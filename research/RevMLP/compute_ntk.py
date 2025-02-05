import copy
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim

from fastbreak.losses import *
from model import RevMLP

from utils import *
from fastbreak.utils import *


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


def compute_all_sims(
    checkpoints, initial_checkpoint, model, input_tensor, device, mode="ntk"
):
    """
    Compute the NTK similarity between all pairs of checkpoints in the given directory.
    """

    init_model = copy.deepcopy(load_checkpoint(initial_checkpoint, model, device=device))
    init_model = init_model.to(device)

    results = {}

    for checkpoint in checkpoints:
        try:
            model = load_checkpoint(checkpoint, model, device=device)
        except:
            continue
        model = model.to(device)

        results[checkpoint] = compute_ntk_similarity(
            init_model, model, input_tensor.to(device), mode=mode
        ).item()

    return results


def combine_dicts_to_csv(dict1, dict2, csv_filename):
    """
    Combine two dictionaries with the same keys into a CSV file.

    :param dict1: First dictionary with "ntk" values.
    :param dict2: Second dictionary with "jacobian" values.
    :param csv_filename: The name of the CSV file to be created.
    """
    # Ensure both dictionaries have the same keys
    if dict1.keys() != dict2.keys():
        raise ValueError("Both dictionaries must have the same keys.")

    # Open the CSV file for writing
    with open(csv_filename, mode="w", newline="") as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)

        # Write the header row
        csv_writer.writerow(["key", "ntk", "jacobian"])

        # Write the rows for each key-value pair
        for key in dict1:
            csv_writer.writerow([key, dict1[key], dict2[key]])


if __name__ == "__main__":
    import glob

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_argparser():
        parser = argparse.ArgumentParser(description="NTK")
        parser.add_argument(
            "--config-path",
            type=str,
            required=True,
            help="Path to the folder containing the config file",
        )
        # parser.add_argument(
        #     "--weight-path",
        #     type=str,
        #     required=True,
        #     help="Path to the folder containing the checkpoints",
        # )
        return parser

    def load_config_file(config_path):
        with open(config_path, "r") as file:
            config_dict = yaml.safe_load(file)

        config_new = {}
        for value in config_dict.values():
            config_new.update(value)
        return argparse.Namespace(**config_new)

    args = create_argparser().parse_args()
    og_folder_path = args.config_path
    if os.path.exists(os.path.join(og_folder_path, "ntk_jac_sims.csv")):
        exit()
    # Check if the provided config path is a directory
    if os.path.isdir(args.config_path):
        # Search for .yaml files in the given directory
        yaml_files = glob.glob(os.path.join(args.config_path, "**/*.yml"), recursive=True)
        if len(yaml_files) == 0:
            raise FileNotFoundError(
                f"No .yaml files found in the directory {args.config_path}"
            )
        elif len(yaml_files) > 1:
            raise RuntimeError(
                f"Multiple .yaml files found in the directory {args.config_path}. Please specify the exact file."
            )
        else:
            # If there is exactly one .yaml file, use it as the config file
            config_path = yaml_files[0]
    elif not os.path.isfile(args.path):
        raise FileNotFoundError(
            f"The specified config file {args.path} does not exist."
        )

    print(load_config_file(config_path))
    parser = Parser(config=load_config_file(config_path))
    (
        exp_args,
        data_args,
        model_args,
        train_args,
        results_args,
    ) = parser.parse()

    print(model_args)

    # Checkpoint handling
    checkpoint_dir = og_folder_path
    if exp_args.train_ggn:
        opt_name = "GGN"
    elif train_args.optimizer["name"] == "adam":
        opt_name = "adam"
    else:
        opt_name = "sgd"
    exp_name = f"{opt_name}_MLP_L{model_args.num_layers}_IB{model_args.hidden_dim}_LR{train_args.optimizer['args']['lr']}_seed{exp_args.seed}"

    timestamp = sorted(list(os.listdir(checkpoint_dir)))[0]
    checkpoint_format = os.path.join(checkpoint_dir, timestamp, "train", "checkpoints", f"{exp_name}_epoch*.pt")

    def get_all_checkpoints(ckpt_format):
        checkpoint_files = glob.glob(ckpt_format)

        for checkpoint_path in checkpoint_files:
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"The specified checkpoint file {checkpoint_path} does not exist."
                )

        return checkpoint_files

    checkpoints = get_all_checkpoints(checkpoint_format)

    initial_checkpoint = checkpoint_format.replace("*", "000")
    if not os.path.isfile(initial_checkpoint):
        raise FileNotFoundError(
            f"The specified checkpoint file {initial_checkpoint} does not exist."
        )
    checkpoints.remove(initial_checkpoint)

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
    model = RevMLP(
        loss_func=loss_func,
        **vars(model_args),
    )
    print(model)

    images, _ = next(iter(train_loader))

    # Assuming model, train_dataloader, test_dataloader, num_classes, device, and block_output_size are defined
    dict_ntk = compute_all_sims(
        checkpoints, initial_checkpoint, model, images, device, mode="ntk"
    )
    dict_jacobian = compute_all_sims(
        checkpoints, initial_checkpoint, model, images, device, mode="jacobian"
    )

    out_path = os.path.join(og_folder_path, "ntk_jac_sims.csv")
    combine_dicts_to_csv(
        dict_ntk,
        dict_jacobian,
        out_path,
    )
    print("Saved to ", out_path)
