import torch
from .loss import FastBreakLossFunction
from typing import Optional

__all__ = ["MSELoss"]


class MSELoss(torch.nn.MSELoss, FastBreakLossFunction):
    """
    Mean Squared Error Loss with support for fast break optimization methods.
    """

    def __init__(self) -> None:
        """
        Initialize the MSELoss.
        """
        super().__init__()

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        U_proj: Optional[torch.Tensor] = None,
        temporal: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for the loss function with optional projection and temporal handling.

        Parameters:
        input (torch.Tensor): Input tensor of predictions.
        target (torch.Tensor): Target tensor of ground truth labels.
        U_proj (Optional[torch.Tensor]): Optional projection matrix to apply to the input.
        temporal (bool): If True, handle input as temporal data.

        Returns:
        torch.Tensor: The computed loss.
        """
        if U_proj is not None:
            if input.dim() == 3 and not temporal:
                input = input.view(input.size(0), -1)
            input = torch.matmul(input, U_proj)
        return super().forward(input, target)

    def compute_pseudoinv_hessian_epsilon(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        U_proj: Optional[torch.Tensor] = None,
        temporal: bool = False,
    ) -> torch.Tensor:
        """
        Compute the product of the pseudo-inverse Hessian and the tangent of the loss function, epsilon.


        Parameters:
        output (torch.Tensor): The output tensor from the model.
        target (torch.Tensor): The ground truth labels.
        U_proj (Optional[torch.Tensor]): Optional projection matrix to apply to the output.
        temporal (bool): If True, handle output as temporal data.

        Returns:
        torch.Tensor: The computed product of the pseudo-inverse Hessian and epsilon.
        """
        og_shape = output.shape
        output_r = output
        if U_proj is not None:
            if output.dim() == 3 and not temporal:
                output_r = output.view(output.size(0), -1)
            output_r = torch.matmul(output_r, U_proj)

        epsilon = 2 * (output_r - target)

        if U_proj is not None:
            # Assuming U_proj is orthogonal, hence the transpose is the pseudo-inverse
            u_pinv = U_proj.t()
            epsilon = torch.matmul(epsilon, u_pinv)
            if not temporal:
                epsilon = epsilon.view(og_shape)  # Reshape to the original output shape

        return epsilon
