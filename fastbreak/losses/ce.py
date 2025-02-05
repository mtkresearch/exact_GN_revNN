import torch
import torch.nn.functional as F
from typing import Optional, Union

# Assuming 'FastBreakLossFunction' is defined in the 'loss' module and imported correctly
from .loss import FastBreakLossFunction

__all__ = ["CrossEntropyLoss"]


class CrossEntropyLoss(torch.nn.CrossEntropyLoss, FastBreakLossFunction):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        size_average: Optional[bool] = None,
        ignore_index: int = -100,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        use_identity: bool = False,
        damping: float = 0.0,
    ) -> None:
        """
        Initialize the CrossEntropyLoss with the possibility of label smoothing.

        Parameters:
        weight (Optional[torch.Tensor]): A manual rescaling weight given to each class.
        size_average (Optional[bool]): Deprecated (see reduction). By default, the losses are averaged over each loss element in the batch.
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.
        reduce (Optional[bool]): Deprecated (see reduction). Indicates whether to reduce the loss.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        label_smoothing (float): If greater than 0, smooth the labels towards 1/n_classes.

        Returns:
        None
        """
        super().__init__(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing, # note: to use this we should also implement the smoothing in the compute_pseudoinv_hessian_epsilon method
        )
        self.use_identity = use_identity
        self.damping = damping

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
        use_identity (bool): If True, the Hessian will be treated as the identity.

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
        if U_proj is not None:
            if output.dim() == 3 and not temporal:
                output = output.reshape(output.shape[0], -1)
            output = output @ U_proj

        if self.use_identity:
            pseudo_inv_hessian_epsilon = torch.nn.Softmax(dim=-1)(output) - target
        else:
            if self.damping != 0.0:
                g = torch.exp(output)
                s = g.sum(dim=1).unsqueeze(1)

                diag_elements = g + (self.damping * s)
                inv_diag_elements = 1 / diag_elements
                k = inv_diag_elements * g
                square = torch.diag_embed(inv_diag_elements * s)
                k_kt = torch.einsum('bi,bj->bij', (k, k)) # batched outer product
                g_k = (g * k).sum(dim=-1)
                second_term = torch.einsum('bij, b->bij', (k_kt, 1/(1 - g_k / s.squeeze())))
                final = square + second_term
                h_plus = final.mean(dim=0)

                epsilon = torch.nn.Softmax(dim=-1)(output) - target
                pseudo_inv_hessian_epsilon = epsilon @ h_plus
            else:
                g = torch.exp(output)
                avg_y_over_g = (1 / target.shape[-1]) * (target / g).sum(dim=-1)
                avg_y_over_g = avg_y_over_g.unsqueeze(dim=-1)
                square_brackets = avg_y_over_g - (target / g)
                pseudo_inv_hessian_epsilon = torch.abs(g).sum(-1, keepdim=True) * square_brackets

                # g = torch.exp(output)
                # avg_y_over_g = (target / g).mean(dim=-1, keepdim=True)
                # square_brackets = avg_y_over_g - (target / g)
                # beta = torch.logsumexp(output, dim=-1, keepdim=True) # (batch_size, 1)
                # pseudo_inv_hessian_epsilon = torch.exp(beta) * square_brackets

        if U_proj is not None:
            # u_pinv = torch.pinverse(U) # TODO: can we just transpose U?
            u_pinv = U_proj.t()
            pseudo_inv_hessian_epsilon = pseudo_inv_hessian_epsilon @ u_pinv
            if not temporal:
                pseudo_inv_hessian_epsilon = pseudo_inv_hessian_epsilon.reshape(
                    og_shape
                )  # for the make_dual, pytorch wants input and tangent of same shape

        return pseudo_inv_hessian_epsilon
