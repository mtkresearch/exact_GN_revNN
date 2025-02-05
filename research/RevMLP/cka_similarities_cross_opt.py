import copy
import os
import re

from fastbreak import ReversibleLinear
from fastbreak.losses import *
from fastbreak.utils import *
from model import RevMLP
from utils import *

device = "cuda"


def load_checkpoint(checkpoint_path, model, device):
    # Ensure that the checkpoint exists
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

    # Load the saved state dict from the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "optimizer" in checkpoint:
        model_ckpt = checkpoint["model"]
        model.load_state_dict(model_ckpt)
    else:
        model.load_state_dict(checkpoint)

    return model

def compute_cka(model_ggn, model_adam, model_sgd, train_loader):
    # compute CKAs
    cka_computer = CudaCKA(device)
    cka_values = {}
    num_blocks_in_model = len(model_ggn.layers)

    # Define a function to be called every time a forward pass is done on a layer
    def get_activation(activations, name):
        def hook(model, input, output):
            x = torch.cat((output[0], output[1]), dim=model.split_dim)
            activations[name] = x.detach()

        return hook

    # Initialize dictionaries to hold the activations for each model
    activations_model_ggn = {}
    activations_model_adam = {}
    activations_model_sgd = {}

    # Register hooks for both models
    hooks_model_ggn = []
    hooks_model_adam = []
    hooks_model_sgd = []
    for name, layer in model_ggn.layers.named_children():
        if isinstance(layer, ReversibleLinear):
            hook = layer.register_forward_hook(
                get_activation(activations_model_ggn, name)
            )
            hooks_model_ggn.append(hook)
    for name, layer in model_adam.layers.named_children():
        if isinstance(layer, ReversibleLinear):
            hook = layer.register_forward_hook(
                get_activation(activations_model_adam, name)
            )
            hooks_model_adam.append(hook)
    for name, layer in model_sgd.layers.named_children():
        if isinstance(layer, ReversibleLinear):
            hook = layer.register_forward_hook(
                get_activation(activations_model_sgd, name)
            )
            hooks_model_sgd.append(hook)

    # Dictionary to store CKA similarities
    cka_similarities = {"ggn_adam": {}, "ggn_sgd": {}, "adam_sgd": {}}

    # Perform forward passes and compute CKA
    with tqdm(train_loader, unit="batch") as tepoch:
        for i, (images, labels) in enumerate(tepoch):
            with torch.no_grad():
                images = images.to(device)
                _ = model_ggn(images)  # Forward pass for model 1
                _ = model_adam(images)  # Forward pass for model 2
                _ = model_sgd(images)  # Forward pass for model 2

                # Compute CKA for each layer
                # print(activations_model_1)
                for layer_name in activations_model_ggn.keys():
                    if layer_name in activations_model_adam.keys():
                        cka_score = (
                            cka_computer.linear_CKA(
                                activations_model_ggn[layer_name].reshape(images.shape[0], -1),
                                activations_model_adam[layer_name].reshape(
                                    images.shape[0], -1
                                ),
                            )
                            .cpu()
                            .item()
                        )
                        cka_similarities["ggn_adam"][layer_name] = cka_score
                    if layer_name in activations_model_sgd.keys():
                        cka_score = (
                            cka_computer.linear_CKA(
                                activations_model_ggn[layer_name].reshape(images.shape[0], -1),
                                activations_model_sgd[layer_name].reshape(
                                    images.shape[0], -1
                                ),
                            )
                            .cpu()
                            .item()
                        )
                        cka_similarities["ggn_sgd"][layer_name] = cka_score
                    if layer_name in activations_model_adam.keys() and layer_name in activations_model_sgd.keys():
                        cka_score = (
                            cka_computer.linear_CKA(
                                activations_model_adam[layer_name].reshape(images.shape[0], -1),
                                activations_model_sgd[layer_name].reshape(
                                    images.shape[0], -1
                                ),
                            )
                            .cpu()
                            .item()
                        )
                        cka_similarities["adam_sgd"][layer_name] = cka_score

    return cka_similarities

def compute_all_ckas(checkpoints_ggn, checkpoints_adam, checkpoints_sgd, models, train_loader, device):
    results = {}

    for checkpoint_ggn, checkpoint_adam, checkpoint_sgd in zip(sorted(checkpoints_ggn), sorted(checkpoints_adam), sorted(checkpoints_sgd)):
        try:
            model_ggn = load_checkpoint(checkpoint_ggn, models[0], device=device)
            model_adam = load_checkpoint(checkpoint_adam, models[1], device=device)
            model_sgd = load_checkpoint(checkpoint_sgd, models[2], device=device)
        except:
            continue
        model_ggn = model_ggn.to(device)
        model_adam = model_adam.to(device)
        model_sgd = model_sgd.to(device)

        match = re.search(r'_epoch(\d+)\.pt', checkpoint_ggn)
        matc2 = re.search(r'_epoch(\d+)\.pt', checkpoint_adam)
        matc3 = re.search(r'_epoch(\d+)\.pt', checkpoint_sgd)
        assert match.group(1) == matc2.group(1), f"{match.group(1)}_{matc2.group(1)}_{checkpoints_ggn}_{checkpoints_adam}"
        assert match.group(1) == matc3.group(1), f"{match.group(1)}_{matc3.group(1)}_{checkpoints_ggn}_{checkpoints_sgd}"

        epoch_number = match.group(1)

        results[int(epoch_number)] = compute_cka(
            model_ggn, model_adam, model_sgd, train_loader
        )

    return results


def save_to_csv(cka_dict, output_path):
    """
    Save the CKA similarities to a CSV file.

    Parameters:
    cka_similarities (dict): CKA similarities of the weights.
    output_path (str): Path to the output CSV file.
    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for couple in ["ggn_adam", "ggn_sgd", "adam_sgd"]:
        out_file = os.path.join(output_path, couple+".csv")
        with open(out_file, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            header = ["Epoch"]
            header.extend([f"cka_layer_{i}" for i in range(len(next(iter(cka_dict[0].values())).keys()))])
            writer.writerow(
                header
            )
            for index, name in enumerate(cka_dict.keys()):
                row_data = [index]
                row_data.extend(list(cka_dict[name][couple].values()))
                writer.writerow(row_data)


if __name__ == "__main__":
    import glob

    def create_argparser():
        parser = argparse.ArgumentParser(description="Linear Classifier Training")
        parser.add_argument(
            "--config",
            type=str,
            required=True,
            help="Path to the folder containing the log folders for all the optimizers",
        )
        parser.add_argument(
            "--seed",
            type=int,
            required=False,
            default=44,
        )
        return parser

    def load_config_file(config_path):
        with open(config_path, "r") as file:
            config_dict = yaml.safe_load(file)

        config_new = {}
        for value in config_dict.values():
            config_new.update(value)
        return argparse.Namespace(**config_new)

    args = create_argparser().parse_args()
    og_folder_path = args.config

    yml_files = []
    for exp_folder in [x for x in os.listdir(og_folder_path) if f"seed{args.seed}" in x and ("GGN" in x or "adam" in x or "sgd" in x)]:
        # Check if the provided config path is a directory
        exp_folder_full_path = os.path.join(og_folder_path, exp_folder)
        if os.path.isdir(exp_folder_full_path):
            # Search for .yaml files in the given directory
            yaml_files = glob.glob(os.path.join(exp_folder_full_path, "**/*.yml"), recursive=True)
            if len(yaml_files) == 0:
                raise FileNotFoundError(
                    f"No .yaml files found in the directory {exp_folder_full_path}"
                )
            elif len(yaml_files) > 1:
                raise RuntimeError(
                    f"Multiple .yaml files found in the directory {exp_folder_full_path}. Please specify the exact file."
                )
            else:
                # If there is exactly one .yaml file, use it as the config file
                yml_files.append(yaml_files[0])
        elif not os.path.isfile(args.config):
            raise FileNotFoundError(
                f"The specified config file {args.config} does not exist."
            )
    yml_files = sorted(yml_files)
    ggn_yaml, adam_yaml, sgd_yaml = yml_files

    print(load_config_file(ggn_yaml))
    parser = Parser(config=load_config_file(ggn_yaml))
    (
        exp_args,
        data_args,
        model_args,
        train_args,
        results_args,
    ) = parser.parse()

    print(model_args)

    # Checkpoint handling
    def get_all_checkpoints(ckpt_format):
        checkpoint_files = glob.glob(ckpt_format)

        for checkpoint_path in checkpoint_files:
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"The specified checkpoint file {checkpoint_path} does not exist."
                )

        return checkpoint_files

    checkpoint_dirs = [x.replace("config.yml", "checkpoints") for x in [ggn_yaml, adam_yaml, sgd_yaml]]
    checkpoint_formats = [f"{opt_name}_MLP_L{model_args.num_layers}_IB{model_args.hidden_dim}_LR*_seed{exp_args.seed}_epoch*.pt" for opt_name in ["GGN", "adam", "sgd"]]
    checkpoints_ggn = get_all_checkpoints(os.path.join(checkpoint_dirs[0], checkpoint_formats[0]))
    checkpoints_adam = get_all_checkpoints(os.path.join(checkpoint_dirs[1], checkpoint_formats[1]))
    checkpoints_sgd = get_all_checkpoints(os.path.join(checkpoint_dirs[2], checkpoint_formats[2]))

    # load model
    LOSSES = {
        "mse": MSELoss,
        "cross_entropy": CrossEntropyLoss,
    }
    loss_func = LOSSES[train_args.loss_function["name"]](
        **train_args.loss_function["args"]
    )

    # load data
    train_data, test_data = load_data(
        data_path=data_args.path,
        dataset=data_args.name.lower(),
        subset=False,
        augmentations=data_args.augmentations,
        output_dim=False,
        num_classes=data_args.num_classes,
        random_proj=None,
    )
    train_loader, test_loader = get_dataloaders(
        train_data=train_data,
        test_data=test_data,
        batch_size=data_args.batch_size,
        drop_last=False,
    )

    models = [RevMLP(loss_func=loss_func,
                     **vars(model_args),) for _ in range(3)]

    # Assuming model, train_dataloader, test_dataloader, num_classes, device, and block_output_size are defined
    cka_similarities = compute_all_ckas(
        checkpoints_ggn,
        checkpoints_adam,
        checkpoints_sgd,
        models,
        train_loader,
        device
    )

    output_path = os.path.join(args.config, f"cka_similarities_cross_opt_seed{args.seed}")
    save_to_csv(cka_similarities, output_path)
    print("Saved to", output_path)
