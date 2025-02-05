import torch
from typing import Optional


def compute_accuracy(
    output: torch.Tensor, target: torch.Tensor, u_proj: Optional[torch.Tensor] = None
) -> float:
    """
    Calculate the accuracy of the output against the target.

    Parameters:
    output (torch.Tensor): The output tensor from the model.
    target (torch.Tensor): The ground truth labels.
    u_proj (Optional[torch.Tensor]): The projection matrix. If provided, it is used to project the output.

    Returns:
    float: The accuracy as a percentage.
    """
    # Reshape output to ensure it is two-dimensional
    output = output.view(output.size(0), -1)

    # Project the output if a projection matrix is provided
    if u_proj is not None:
        output = torch.matmul(output, u_proj)

    # Calculate the accuracy
    correct_predictions = (target.argmax(dim=-1) == output.argmax(dim=-1)).sum()
    accuracy = 100.0 * correct_predictions / target.size(0)

    return accuracy
