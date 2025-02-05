from abc import ABC, abstractmethod
import torch

__all__ = ["FastBreakLossFunction"]


class FastBreakLossFunction(ABC):
    """
    Abstract base class for loss functions that support fast break optimization methods.
    """

    def __init__(self) -> None:
        """
        Initialize the FastBreakLossFunction.
        """
        super().__init__()

    @abstractmethod
    def compute_pseudoinv_hessian_epsilon(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the product of the pseudo-inverse Hessian and the tangent of the loss function, epsilon.

        Parameters:
        output (torch.Tensor): The output tensor from the model.
        target (torch.Tensor): The ground truth labels.

        Returns:
        torch.Tensor: The computed product of the pseudo-inverse Hessian and epsilon.
        """
        raise NotImplementedError("This method needs to be implemented by subclasses.")
