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

def compute_cka(model_1, model_2, train_loader):
    # compute CKAs
    cka_computer = CudaCKA(device)
    cka_values = {}
    num_blocks_in_model = len(model_1.layers)

    # Define a function to be called every time a forward pass is done on a layer
    def get_activation(activations, name):
        def hook(model, input, output):
            x = torch.cat((output[0], output[1]), dim=model.split_dim)
            activations[name] = x.detach()

        return hook

    # Initialize dictionaries to hold the activations for each model
    activations_model_1 = {}
    activations_model_2 = {}

    # Register hooks for both models
    hooks_model_1 = []
    hooks_model_2 = []
    for name, layer in model_1.layers.named_children():
        if isinstance(layer, ReversibleLinear):
            hook = layer.register_forward_hook(
                get_activation(activations_model_1, name)
            )
            hooks_model_1.append(hook)
    for name, layer in model_2.layers.named_children():
        if isinstance(layer, ReversibleLinear):
            hook = layer.register_forward_hook(
                get_activation(activations_model_2, name)
            )
            hooks_model_2.append(hook)

    # Dictionary to store CKA similarities
    cka_similarities = {}

    # Perform forward passes and compute CKA
    with tqdm(train_loader, unit="batch") as tepoch:
        for i, (images, labels) in enumerate(tepoch):
            with torch.no_grad():
                images = images.to(device)
                _ = model_1(images)  # Forward pass for model 1
                _ = model_2(images)  # Forward pass for model 2

                # Compute CKA for each layer
                # print(activations_model_1)
                for layer_name in activations_model_1.keys():
                    if layer_name in activations_model_2.keys():
                        cka_score = (
                            cka_computer.linear_CKA(
                                activations_model_1[layer_name].reshape(images.shape[0], -1),
                                activations_model_2[layer_name].reshape(
                                    images.shape[0], -1
                                ),
                            )
                            .cpu()
                            .item()
                        )
                        cka_similarities[layer_name] = cka_score

    return cka_similarities

def compute_all_ckas(checkpoints, initial_checkpoint, model, train_loader, device):
    init_model = copy.deepcopy(load_checkpoint(initial_checkpoint, model, device=device))
    init_model = init_model.to(device)

    results = {}

    for checkpoint in checkpoints:
        try:
            model = load_checkpoint(checkpoint, model, device=device)
        except:
            continue
        model = model.to(device)

        match = re.search(r'_epoch(\d+)\.pt', checkpoint)

        epoch_number = match.group(1)

        results[int(epoch_number)] = compute_cka(
            init_model, model, train_loader
        )

    return results


def save_to_csv(cka_dict, output_path):
    """
    Save the CKA similarities to a CSV file.

    Parameters:
    cka_similarities (dict): CKA similarities of the weights.
    output_path (str): Path to the output CSV file.
    """
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["Epoch"]
        header.extend([f"cka_layer_{i}" for i in range(len(next(iter(cka_dict.values())).keys()))])
        writer.writerow(
            header
        )
        for index, name in enumerate(cka_dict.keys()):
            row_data = [index]
            row_data.extend(list(cka_dict[name].values()))
            writer.writerow(row_data)


if __name__ == "__main__":
    import glob

    def create_argparser():
        parser = argparse.ArgumentParser(description="CKA")
        parser.add_argument(
            "--config",
            type=str,
            required=True,
            help="Path to the folder containing the config file",
        )
        # parser.add_argument(
            # "--weight-path",
            # type=str,
            # required=True,
            # help="Path to the folder containing the checkpoints",
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
    og_folder_path = args.config
    # if os.path.exists(os.path.join(og_folder_path, "cka_similarities.csv")):
        # exit()
    # Check if the provided config path is a directory
    if os.path.isdir(args.config):
        # Search for .yaml files in the given directory
        yaml_files = glob.glob(os.path.join(args.config, "**/*.yml"), recursive=True)
        if len(yaml_files) == 0:
            raise FileNotFoundError(
                f"No .yaml files found in the directory {args.config}"
            )
        elif len(yaml_files) > 1:
            raise RuntimeError(
                f"Multiple .yaml files found in the directory {args.config}. Please specify the exact file."
            )
        else:
            # If there is exactly one .yaml file, use it as the config file
            args.config = yaml_files[0]
    elif not os.path.isfile(args.config):
        raise FileNotFoundError(
            f"The specified config file {args.config} does not exist."
        )

    print(load_config_file(args.config))
    parser = Parser(config=load_config_file(args.config))
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
        print(checkpoint_files)

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

    # Assuming model, train_dataloader, test_dataloader, num_classes, device, and block_output_size are defined
    cka_similarities = compute_all_ckas(
        checkpoints,
        initial_checkpoint,
        model,
        train_loader,
        device
    )

    output_csv_path = os.path.join(og_folder_path, "cka_similarities_new.csv")
    # output_csv_path = os.path.join(os.path.dirname(args.config), "cka_similarities.csv")
    save_to_csv(cka_similarities, output_csv_path)
    print("Saved to", output_csv_path)
