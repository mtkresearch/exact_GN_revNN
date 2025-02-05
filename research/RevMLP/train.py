import numpy as np
import os
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

from fastbreak.utils import (
    get_U_matrix,
    set_seed,
    get_synthetic_data,
    load_data,
    get_dataloaders,
)
from fastbreak.losses import *
from fastbreak.utils import *

from model.reversible_mlp import RevMLP

from utils import *


LOSSES = {
    "mse": MSELoss,
    "cross_entropy": CrossEntropyLoss,
}

SCHEDULERS = {"step": StepLR}

OPTIMIZERS = {"adam": optim.Adam, "sgd": optim.SGD}

MODELS = {"reversible": RevMLP}

torch.set_default_dtype(torch.float32)


def main_train(args, tuning=False):
    # Parse the arguments.
    parser = Parser(config=args)

    (
        exp_args,
        data_args,
        model_args,
        train_args,
        results_args,
    ) = parser.parse()

    set_seed(exp_args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and exp_args.device else "cpu"
    )

    if exp_args.train_ggn:
        opt_name = "GGN"
    elif train_args.optimizer["name"] == "adam":
        opt_name = "adam"
    else:
        opt_name = "sgd"
    exp_name = f"MLP_L{model_args.num_layers}_IB{model_args.hidden_dim}_LR{train_args.optimizer['args']['lr']}_seed{exp_args.seed}"

    out_dir = configure_logger(
        exp_args, data_args, model_args, train_args, results_args, prefix_path="train", name=f"{opt_name}_{exp_name}"
    )

    # setuip weight directory
    weight_root_dir = os.path.join(out_dir, "train", "checkpoints")
    if not os.path.exists(weight_root_dir):
        os.mkdir(weight_root_dir)

    if data_args.name == "synthetic":
        logger.log("Creating a synthetic dataset...")
        if data_args.num_channels > 0:
            x = torch.randn(
                data_args.batch_size, data_args.num_channels, data_args.input_size
            )
            y = torch.randn(
                data_args.batch_size, data_args.num_channels, data_args.output_size
            )
        else:
            x = torch.randn(data_args.batch_size, data_args.input_size)
            y = torch.randn(data_args.batch_size, data_args.output_size)

        train_data = torch.utils.data.TensorDataset(x, y)
        test_data = torch.utils.data.TensorDataset(x, y)

        if train_args.loss_function["name"].lower() == "mse":
            logger.log("Creating the synthetic targets...")
            target = torch.ones_like(x) * 3
            target_prev = target.clone()
            target = target.to(device)
            target_prev = target_prev.to(device)

    else:
        rand_proj = False
        if data_args.random_proj:
            rand_proj = get_U_matrix(
                data_args.args["num_labels"],
                model_args.feature_dim,
                device="cpu",
            )
        train_data, test_data = load_data(
            data_path=data_args.path,
            dataset=data_args.name.lower(),
            subset=data_args.subset,
            augmentations=data_args.augmentations,
            output_dim=False,
            num_classes=data_args.num_classes,
            random_proj=rand_proj,
        )

    logger.log("Creating the dataloaders...")
    train_loader, test_loader = get_dataloaders(
        train_data,
        test_data,
        data_args.batch_size,
        drop_last=False,
    )

    loss_func = LOSSES[train_args.loss_function["name"]](
        **train_args.loss_function.get("args", {})
    )

    logger.log("Initialising the model...")
    model = MODELS["reversible" if exp_args.reversible else "invertible"](
        loss_func=loss_func, **vars(model_args)
    )
    model = model.to(device)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0, std=1e-3)

    model.apply(init_weights)

    if train_args.checkpoint_path is not None:
        # Load the model
        model = load_model(train_args.checkpoint_path, model, device=device)

    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Number of parameters: {num_params}")
    print(f"Number of parameters: {num_params}")

    if train_args.Task["name"].lower() == "labeller":
        labeller = MODELS["reversible" if exp_args.reversible else "invertible"](
            **vars(model_args)
        )
        labeller = labeller.to(device)

        labeller.apply(init_weights)
        model.apply(init_weights)
    else:
        labeller = None

    optimizer = OPTIMIZERS[train_args.optimizer["name"].lower()](
        model.parameters(), **train_args.optimizer["args"]
    )
    if train_args.checkpoint_path is not None:
        optimizer = load_optimizer(train_args.checkpoint_path, optimizer, device=device)

    if train_args.scheduler:
        scheduler = SCHEDULERS[train_args.scheduler["name"].lower()](
            optimizer, **train_args.scheduler["args"]
        )
    else:
        scheduler = None

    train(
        model,
        train_loader,
        test_loader,
        loss_func,
        optimizer,
        scheduler,
        device,
        logger=logger,
        labeller=labeller,
        classification=(
            True if train_args.Task["name"].lower() == "classification" else False
        ),
        train_ggn=exp_args.train_ggn,
        added_random_in_output=data_args.add_random,
        random_proj_output=data_args.random_proj,
        rand_proj_matrix=rand_proj.to(device) if data_args.random_proj else None,
        tuning=tuning,
        weight_folder=weight_root_dir,
        exp_name=exp_name,
        opt_name=opt_name,
        **train_args.args,
    )

    print("Logged to:", out_dir)


def configure_logger(
    exp_args, data_args, model_args, train_args, results_args, prefix_path, name="",
):

    output_dir = os.path.join(
        results_args.output_dir,
        name,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    print("saving to: \n", output_dir)

    full_path = os.path.join(output_dir, prefix_path)

    logger.configure(
        full_path,
        format_strs=results_args.format_strs,
        hps={
            "Setup": {
                "Experiment": vars(exp_args),
                "Data": vars(data_args),
                "Model": vars(model_args),
            },
            "Runtime": {
                "Train": vars(train_args),
            },
            "Output": {
                "Results": vars(results_args),
            },
        },
    )
    print("logging to", output_dir)
    return output_dir


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args = create_argparser().parse_args()
    main_train(args, tuning=False)
