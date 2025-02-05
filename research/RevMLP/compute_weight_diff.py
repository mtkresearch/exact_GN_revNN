import torch
import csv
import glob
import os
import sys
import torch.nn.functional as F


def load_checkpoint(filename):
    """
    Load a model checkpoint from a specified directory.

    Parameters:
    directory (str): Directory containing the model checkpoints.
    filename (str): Filename of the model checkpoint.

    Returns:
    dict: Model checkpoint.
    """
    checkpoint_path = filename
    ckpt = torch.load(checkpoint_path)
    if "optimizer" in ckpt:
        return ckpt["model"]
    else:
        return ckpt


def compute_weight_norms(model_checkpoint):
    """
    Compute the L2 norms of the weights in a model checkpoint.

    Parameters:
    model_checkpoint (dict): Model checkpoint.

    Returns:
    dict: L2 norms of the weights for each layer.
    """
    norms = {}
    for name, parameters in model_checkpoint.items():
        norms[name] = torch.norm(parameters, p=2).item()
    return norms


def calculate_relative_difference(norms1, norms2):
    """
    Calculate the relative difference in norms between two sets of norms.

    Parameters:
    norms1 (dict): L2 norms of the weights for the first checkpoint.
    norms2 (dict): L2 norms of the weights for the second checkpoint.

    Returns:
    dict: Relative differences in norms.
    """
    relative_differences = {}
    for name in norms1:
        if name in norms2:
            if norms1[name] != 0:
                relative_differences[name] = norms2[name] / norms1[name]
    return relative_differences


def compute_cosine_similarity(checkpoint1, checkpoint2):
    """
    Compute the cosine similarity between the weights of two checkpoints.

    Parameters:
    checkpoint1 (dict): First model checkpoint.
    checkpoint2 (dict): Second model checkpoint.

    Returns:
    dict: Cosine similarity for each layer.
    """
    cosine_similarities = {}
    for name, parameters in checkpoint1.items():
        if name in checkpoint2:
            similarity = F.cosine_similarity(
                parameters.flatten().unsqueeze(0),
                checkpoint2[name].flatten().unsqueeze(0),
            )
            cosine_similarities[name] = similarity.item()
    return cosine_similarities


def save_to_csv(relative_differences, cosine_similarities, output_path):
    """
    Save the relative differences and cosine similarities to a CSV file.

    Parameters:
    relative_differences (dict): Relative differences in norms.
    cosine_similarities (dict): Cosine similarities of the weights.
    output_path (str): Path to the output CSV file.
    """
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["Layer Index", "Layer Name", "Relative Difference", "Cosine Similarity"]
        )
        for index, name in enumerate(relative_differences.keys()):
            writer.writerow(
                [
                    index,
                    name,
                    relative_differences[name],
                    cosine_similarities.get(name, ""),
                ]
            )

if __name__ == "__main__":
    directory_path = sys.argv[1]
    if os.path.exists(os.path.join(directory_path, "relative_differences.csv")):
        exit()
    timestamp = list(os.listdir(directory_path))[0]
    checkpoint_format = os.path.join(directory_path, timestamp, "train", "checkpoints", f"*epoch*.pt")
    all_checkpoints = glob.glob(checkpoint_format)
    first_ckpt = [x for x in all_checkpoints if "000" in x][0]
    last_ckpt = [x for x in all_checkpoints if "100" in x][0]

    checkpoint_best = load_checkpoint(first_ckpt)
    checkpoint_initial = load_checkpoint(last_ckpt)

    norms_best = compute_weight_norms(checkpoint_best)
    norms_initial = compute_weight_norms(checkpoint_initial)

    relative_differences = calculate_relative_difference(norms_initial, norms_best)
    cosine_similarities = compute_cosine_similarity(checkpoint_best, checkpoint_initial)

    # Save the relative differences and cosine similarities to a CSV file
    output_csv_path = os.path.join(directory_path, "relative_differences.csv")
    save_to_csv(relative_differences, cosine_similarities, output_csv_path)

    print(f"Relative differences have been saved to {output_csv_path}")
