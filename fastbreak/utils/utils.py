from functools import partial
import numpy as np
import os
import random
import torch
from torchvision import transforms, datasets
import collections.abc
from itertools import repeat
from functools import partial

from .data_ops import *
from .rsvd import *


import torch
from scipy.sparse.linalg import gmres


def apply_threshold(U, S, Vt, rtol, atol=0.0):
    if atol is None:
        atol = 0.0
    if rtol is None:
        rtol = 0.0

    idx = ((S >= S.max() * rtol) & (S >= atol)).nonzero(as_tuple=True)[0]

    # Truncate the matrices based on the threshold
    U_truncated = U[:, idx]
    S_truncated = S[idx]
    Vt_truncated = Vt[idx, :]
    return U_truncated, S_truncated, Vt_truncated


def gmres_solver(A, b, tol=1e-5, max_iter=1000):
    A_np = A.cpu().detach().numpy()
    b_np = b.cpu().detach().numpy()
    x, exitCode = gmres(A_np, b_np, tol=tol, maxiter=max_iter)
    return torch.tensor(x, dtype=torch.float32, device=A.device)


def damped_least_squares(a, damping_factor=1e-3):
    ata = a.t() @ a
    damping_factor = torch.linalg.svdvals(ata).max() * damping_factor
    damping_matrix = damping_factor * torch.eye(ata.size(0)).to(ata.device)
    damped = ata + damping_matrix
    damped_inv = torch.inverse(damped)
    return damped_inv @ a.t()


def compute_pinverse(
    input,
    method="default",
    method_kwargs=None,
):
    if method == "tp":
        out_rev_plus = input.transpose(0, 1)
        return out_rev_plus

    elif method == "qr":
        Q, R = torch.linalg.qr(input, "reduced")
        R_inv = torch.pinverse(R)
        out_rev_plus = R_inv @ Q.t()
        return out_rev_plus

    elif method == "damped":
        out_rev_plus = damped_least_squares(input, damping_factor=method_kwargs["rtol"])
        return out_rev_plus

    elif method == "fast-rsvd":
        # call torch.svd_lowrank (alg 4.4 in Halko's paper)
        U, S, Vt = fast_rsvd(input, method_kwargs)

    elif method == "adaptive-rsvd":
        # adaptive range finder (alg 4.2 in Halko's paper)
        U, S, Vt = adaptive_rsvd(input, method_kwargs)

    elif method == "default":
        # Normal pseudo inverse
        U, S, Vt = torch.linalg.svd(input, full_matrices=False)
    else:
        raise NotImplementedError(f"Method {method} is not currently implemented.")

    if method_kwargs["rtol"] or method_kwargs["atol"]:
        # apply threshold on svd. this gives extra safety to avoid numerical issues
        U, S, Vt = apply_threshold(
            U=U, S=S, Vt=Vt, rtol=method_kwargs["rtol"], atol=method_kwargs["atol"]
        )

    out_rev_plus = Vt.T @ torch.div(U, S).T
    return out_rev_plus


def compute_ggn_update(
    input,
    jvp,
    batch_size=None,
    d=None,
    method="default",
    method_kwargs=None,
):
    if method == "tp":
        out_rev_plus = input.transpose(0, 1)
        return out_rev_plus @ jvp

    elif method == "fast-rsvd":
        # call torch.svd_lowrank (alg 4.4 in Halko's paper)
        U, S, Vt = fast_rsvd(input, method_kwargs)

    elif method == "adaptive-rsvd":
        # adaptive range finder (alg 4.2 in Halko's paper)
        U, S, Vt = adaptive_rsvd(input, method_kwargs)

    elif method == "default":
        # Normal pseudo inverse
        U, S, Vt = torch.linalg.svd(input, full_matrices=False)
    else:
        raise NotImplementedError(f"Method {method} is not currently implemented.")

    if method_kwargs["rtol"] or method_kwargs["atol"]:
        # apply threshold on svd. this gives extra safety to avoid numerical issues
        U, S, Vt = apply_threshold(
            U=U, S=S, Vt=Vt, rtol=method_kwargs["rtol"], atol=method_kwargs["atol"]
        )

    out_rev_plus = Vt.T @ torch.div(U, S).T
    return out_rev_plus @ jvp


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def load_model(checkpoint_path, model, device=torch.device("cpu")):
    """
    Loads the model weights from a given checkpoint path.

    Parameters:
    - checkpoint_path: The path to the checkpoint file.
    - model: The model instance with the same architecture as the saved model.

    Returns:
    - model: The model loaded with weights from the checkpoint.
    """
    # Ensure that the checkpoint exists
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

    # Load the saved state dict from the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load the state dict into the model
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    return model


def load_optimizer(checkpoint_path, optimizer, device=torch.device("cpu")):
    """
    Loads the model weights from a given checkpoint path.

    Parameters:
    - checkpoint_path: The path to the checkpoint file.
    - model: The model instance with the same architecture as the saved model.

    Returns:
    - model: The model loaded with weights from the checkpoint.
    """
    # Ensure that the checkpoint exists
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

    # Load the saved state dict from the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load the state dict into the model
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return optimizer


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def vec(matrix):
    num_cols = matrix.shape[1]
    cols = [matrix[:, i] for i in range(num_cols)]
    return torch.cat(cols, 0)


def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def get_U_matrix(model_out_dim, final_dim, device, fast=False, scipy=False):
    if fast:
        A = torch.randn(model_out_dim, final_dim, device=device)
        A = A / (torch.sqrt(torch.tensor(model_out_dim)))
        return A

    if final_dim >= 512:
        X = torch.normal(0, 1, (final_dim, model_out_dim))
        if scipy:
            import scipy

            Q = torch.linalg.orth(X)
            orth = torch.tensor(Q, dtype=torch.float, device=device)
        else:
            Q, _ = torch.linalg.qr(X, mode="complete")
            orth = Q.to(device=device)
    else:
        dim = max(model_out_dim, final_dim)
        gaus = torch.randn(dim, dim, device=device)
        svd = torch.linalg.svd(gaus)
        orth = svd[0] @ svd[2]
    orth = orth[:model_out_dim, :final_dim]
    return orth


def get_synthetic_data(
    num_images=10000,
    image_size=32,
    correlated=True,
    num_channels=1,
    num_labels=10,
    model_type="dense",
    multiplier=False,
):
    if model_type == "dense":
        assert num_channels == 1
    print("Generating fake data")
    if correlated:
        if image_size <= 100:
            # dim = (image_size**2) * num_channels
            dim = image_size**2
            A = torch.randn(dim, dim)
            q, r = torch.linalg.qr(A)  # , mode="complete")
            q = q * torch.sign(torch.diagonal(r))
            d = torch.tensor([1 / (i**2) for i in range(1, dim + 1)])
            sigma = q * d @ q.transpose(0, 1)
            distrib = torch.distributions.MultivariateNormal(
                torch.zeros(dim), sigma, validate_args=False
            )
            fake_images = distrib.sample([num_images * num_channels]) * 200
            fake_test_images = distrib.sample([(num_images // 10) * num_channels])
            # fake_images = distrib.sample([num_images]) #* 5
            # fake_test_images = distrib.sample([(num_images//10)])

            if multiplier:
                div = 1
                multiplier = torch.arange((image_size**2) // div)
                multiplier = multiplier.repeat(div)
                m = torch.ones(image_size**2)
                m[: multiplier.shape[0]] = multiplier
                perm = torch.randperm(m.shape[0])
                m = m[perm]
                m = torch.unsqueeze(m, 0)
                fake_images = fake_images * m
        else:
            train_data_file = os.path.join(
                "data", f"fakeimages{model_type}_{image_size**2}.pt"
            )
            test_data_file = os.path.join(
                "data", f"faketestimages{model_type}_{image_size**2}.pt"
            )
            if os.path.exists(train_data_file):
                print("Loading synthetic data form file", train_data_file)
                fake_images = torch.load(train_data_file)
                print("Loading synthetic data form file", test_data_file)
                fake_test_images = torch.load(test_data_file)
            else:
                dim = image_size**2
                # dim = (image_size**2) * num_channels
                A = torch.randn(dim, dim) / torch.sqrt(torch.tensor(dim))
                d = torch.sqrt(torch.tensor([1 / (i**2) for i in range(1, dim + 1)]))
                # B = torch.randn(dim, num_images)
                B = torch.randn(dim, num_images * num_channels)
                fake_images = ((A * d) @ B).transpose(0, 1)
                torch.save(fake_images, train_data_file)
                # B = torch.randn(dim, (num_images//10))
                B = torch.randn(image_size**2, (num_images // 10) * num_channels)
                fake_test_images = ((A * d) @ B).transpose(0, 1)
                torch.save(fake_test_images, test_data_file)
                print("Saved synthetic data in", train_data_file, test_data_file)

        if model_type == "conv":
            fake_images = reshape_fortran(
                fake_images, (num_images, num_channels, image_size, image_size)
            )
            fake_test_images = reshape_fortran(
                fake_test_images,
                (num_images // 10, num_channels, image_size, image_size),
            )
        elif model_type == "dense":
            # print(fake_images.shape)
            # torch.set_printoptions(threshold=10_000)
            # from test_training_conv_guillaume import rand_svd
            # _, s, _ = rand_svd(fake_images, int(fake_images.shape[1]*0.2), 50, power=3)
            # s = torch.sort(s, descending=True)
            # print("input", s[0][:20])

            # import matplotlib.pyplot as plt
            # plt.matshow(fake_images[:50, :100].detach().cpu().numpy())
            # plt.savefig("fig_in.png", dpi=300)
            # exit()

            fake_images = fake_images.reshape(num_images, image_size, image_size)
            fake_test_images = fake_test_images.reshape(
                num_images // 10, image_size, image_size
            )
    else:
        fake_images = torch.randn(num_images, num_channels, image_size, image_size)
        fake_test_images = torch.randn(
            num_images // 10, num_channels, image_size, image_size
        )

    fake_labels = torch.randn(num_images, num_labels)
    fake_test_labels = torch.randn(num_images // 10, num_labels)

    train_dataset = torch.utils.data.TensorDataset(fake_images, fake_labels)
    test_dataset = torch.utils.data.TensorDataset(fake_test_images, fake_test_labels)
    print("Finished generating fake data")
    return train_dataset, test_dataset


def load_data(
    data_path,
    dataset,
    subset=False,
    output_dim=False,
    random_proj=False,
    augmentations=None,
    num_classes=10,
):
    # code from https://github.com/jeonsworld/MLP-Mixer-Pytorch/blob/main/utils/data_utils.py
    aug_dict = {
        "randomresizecrop": partial(transforms.RandomResizedCrop, antialias=True),
        "randomhorizontalflip": transforms.RandomHorizontalFlip,
        "noise": AddGaussianNoise,
        "cifarautoaugment": CIFAR10Policy,
    }
    if dataset == "cifar":
        transform_train = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.49139968, 0.48215827, 0.44653124],
                std=[0.24703233, 0.24348505, 0.26158768],
            ),
        ]
        if augmentations:
            for aug_type in augmentations:
                if aug_type["name"].lower() == "cifarautoaugment":
                    # Autoaugment needs to be before ToTensor()
                    transform_train.insert(0, CIFAR10Policy())
                else:
                    transform_train.append(
                        aug_dict[aug_type["name"].lower()](**aug_type.get("args", {}))
                    )

        transform_train = transforms.Compose(transform_train)
        transform_test = transforms.Compose(
            [
                # transforms.CenterCrop((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.49139968, 0.48215827, 0.44653124],
                    std=[0.24703233, 0.24348505, 0.26158768],
                ),
            ]
        )
    elif dataset == "mnist":
        transform_train = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
        if augmentations:
            for aug_type in augmentations:
                transform_train.append(
                    aug_dict[aug_type["name"].lower()](**aug_type.get("args", {}))
                )

        transform_train = transforms.Compose(transform_train)
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    else:
        NotImplementedError(f"{dataset} dataset implemented")

    if isinstance(random_proj, torch.Tensor):
        rand_ort = random_proj
        noise_level = 0.0
        target_transform = transforms.Lambda(
            lambda y: rand_ort[y, :] + torch.randn(rand_ort[y, :].size()) * noise_level
        )
    elif output_dim:
        block_size = output_dim // num_classes
        target_transform = transforms.Lambda(
            lambda y: torch.cat(
                (
                    torch.zeros(num_classes * block_size).scatter_(
                        dim=0,
                        index=torch.arange(y * block_size, (y + 1) * block_size),
                        value=1,
                    ),
                    torch.zeros(output_dim % num_classes),
                )
            )
        )
    else:
        target_transform = transforms.Lambda(
            lambda y: torch.zeros(num_classes).scatter_(0, torch.tensor(y), value=1)
        )

    if dataset == "cifar":
        train_data = datasets.CIFAR10(
            root=data_path,
            train=True,
            download=True,
            transform=transform_train,
            target_transform=target_transform,
        )
        test_data = datasets.CIFAR10(
            root=data_path,
            train=False,
            download=True,
            transform=transform_test,
            target_transform=target_transform,
        )
    elif dataset == "mnist":
        train_data = datasets.MNIST(
            root=data_path,
            train=True,
            download=True,
            transform=transform_train,
            target_transform=target_transform,
        )
        test_data = datasets.MNIST(
            root=data_path,
            train=False,
            download=True,
            transform=transform_test,
            target_transform=target_transform,
        )
    else:
        NotImplementedError(f"{dataset} not implemented.")

    if subset:
        indices = torch.arange(subset)
        train_data = torch.utils.data.Subset(train_data, indices)

        return train_data, test_data
    else:
        return train_data, test_data


def get_dataloaders(
    train_data,
    test_data,
    batch_size,
    drop_last,
    num_workers=1,
):
    loaders = {
        "train": torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            num_workers=num_workers,
        ),
        "test": torch.utils.data.DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
    return loaders["train"], loaders["test"]


def get_dense_teacher_network(
    in_dim, out_dim, num_layers, output_per_layer=False, simplified=False
):
    class LabelerNet(torch.nn.Module):
        def __init__(self, in_dim, out_dim, num_layers=2):
            super(LabelerNet, self).__init__()
            self.sigma = torch.nn.ReLU()
            # self.sigma = torch.nn.Identity()
            self.linear1 = torch.nn.Linear(in_dim, out_dim, bias=simplified == False)
            layers = []
            for _ in range(num_layers - 1):
                layers.append(
                    torch.nn.Linear(out_dim, out_dim, bias=simplified == False)
                )
            self.layers = torch.nn.ModuleList(layers)
            self.output_per_layer = output_per_layer

        def forward(self, x):
            layer_outputs = []
            x = self.linear1(x)
            x = self.sigma(x)
            layer_outputs.append(torch.clone(x))
            for l in self.layers:
                x = l(x)
                x = self.sigma(x)
                layer_outputs.append(torch.clone(x))
            if self.output_per_layer:
                return x, layer_outputs
            else:
                return x

    return LabelerNet(in_dim, out_dim, num_layers)


def get_simple_ddn_teacher_network(
    in_dim, out_dim, num_layers, output_per_layer=False, simplified=False
):
    class LabelerNet(torch.nn.Module):
        def __init__(self, in_dim, out_dim, num_layers=2):
            super(LabelerNet, self).__init__()
            self.sigma = torch.nn.ReLU()
            self.linear1 = torch.nn.Linear(in_dim, out_dim, bias=simplified == False)
            layers = []
            for _ in range(num_layers - 1):
                layers.append(
                    torch.nn.Linear(out_dim, out_dim, bias=simplified == False)
                )
            self.layers = torch.nn.ModuleList(layers)
            self.output_per_layer = output_per_layer

        def forward(self, x):
            layer_outputs = []
            x = self.linear1(x)
            layer_outputs.append(torch.clone(x))
            for l in self.layers:
                x = self.sigma(x)
                x = l(x)
                layer_outputs.append(torch.clone(x))
            if self.output_per_layer:
                return x, layer_outputs
            else:
                return x

    return LabelerNet(in_dim, out_dim, num_layers)


def get_conv_teacher_netowork(
    in_channels, out_channels, kernel_size, num_layers, style="conv"
):
    class LabelerNetConv(torch.nn.Module):
        def __init__(
            self, in_channels, out_channels, kernel_size, num_layers=2, style="conv"
        ):
            super(LabelerNetConv, self).__init__()
            # self.sigma = torch.nn.Identity()
            # self.sigma = torch.nn.Tanh()
            self.sigma = torch.nn.ReLU()
            if style == "conv":
                self.conv1 = torch.nn.Conv1d(
                    in_channels, out_channels, kernel_size, padding="same"
                )
            else:
                self.conv1 = torch.nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=(kernel_size - 1) // 2,
                )
            layers = []
            for _ in range(num_layers - 1):
                if style == "conv":
                    layers.append(
                        torch.nn.Conv1d(
                            out_channels, out_channels, kernel_size, padding="same"
                        )
                    )
                else:
                    layers.append(
                        torch.nn.ConvTranspose1d(
                            out_channels,
                            out_channels,
                            kernel_size,
                            padding=(kernel_size - 1) // 2,
                        )
                    )
            self.layers = torch.nn.ModuleList(layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.sigma(x)
            for l in self.layers:
                x = l(x)
                x = self.sigma(x)
                # x_l = l(x)
                # x = self.sigma(x) + x_l
            return x

    return LabelerNetConv(in_channels, out_channels, kernel_size, num_layers, style)


def get_dense_deq_teacher(G_func, in_dim, hidden_dim, f_thresh, solver, style="same"):
    class DEQDense(torch.nn.Module):
        def __init__(self, G_func, in_dim, hidden_dim, f_thresh, solver, style):
            super(DEQDense, self).__init__()
            self.hidden_dim = hidden_dim
            self.f = G_func(in_dim, hidden_dim)
            self.solver = solver
            self.f_thresh = f_thresh
            self.style = style

        def forward(self, x):
            batch_size = x.shape[0]

            if self.style == "same":
                # compute forward pass and re-engage autograd tape
                with torch.no_grad():
                    z = self.solver(
                        lambda z: self.f(x, z),
                        torch.zeros(batch_size, self.hidden_dim, device=x.device),
                        threshold=self.f_thresh,
                    )
                    z = z.squeeze()
                return z

            elif self.style == "per-layer":
                z = [torch.zeros(batch_size, self.hidden_dim, device=x.device)]
                for _ in range(self.f_thresh):
                    z.append(self.f(x, z[-1]))
                return z[-1], z[1:]

    return DEQDense(G_func, in_dim, hidden_dim, f_thresh, solver, style)
