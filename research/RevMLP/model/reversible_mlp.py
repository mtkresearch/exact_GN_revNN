from functools import partial
import torch
from torch import nn as nn
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F

from fastbreak import MSELoss, ReversibleMLP


__all__ = ["RevMLP"]


class RevMLP(ReversibleMLP):
    def __init__(
        self,
        feature_dim,
        num_layers,
        hidden_dim=None,
        output_dim=None,
        bias=False,
        layer_norm=False,
        non_linearity=False,
        loss_func=MSELoss(),
        inv_method="default",
        inv_method_args=None,
        rtol=0.0,
        atol=1e-7,
    ) -> None:
        super(RevMLP, self).__init__(
            feature_dim,
            num_layers,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            bias=bias,
            layer_norm=(
                partial(
                    torch.nn.LayerNorm,
                    eps=1e-6,
                    elementwise_affine=False,  # , bias=False
                )
                if layer_norm
                else None
            ),
            non_linearity=nn.ReLU() if non_linearity else nn.Identity(),
            inv_method=inv_method,
            inv_method_args=inv_method_args,
            rtol=rtol,
            atol=atol,
        )

        self.loss_func = loss_func

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)

        x = super().forward(x)

        return x

    def backward(self, out, label, logger=None):
        """We always save outputs and targets in the same 'dimension order' as the input (usually, (batch_size, num_channels, feature_dim)).
        Similarly, we expect 'out' to be of same dimension order as the input."""
        pseudo_inv_hessian_epsilon = self.loss_func.compute_pseudoinv_hessian_epsilon(
            output=out,
            target=label,
            U_proj=self.u_proj,
        )

        with fwAD.dual_level():
            x_prev = fwAD.make_dual(tensor=out, tangent=pseudo_inv_hessian_epsilon)

            super().backward(x_prev, logger=logger)

        return x_prev

    def forward_verbose(self, x):
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)

        outs = [x]

        x1, x2 = torch.chunk(x, 2, dim=-1)

        for layer in self.layers:
            x1, x2 = layer.forward(x1, x2)
            outs.append(torch.cat((x1, x2), dim=-1))

        return outs

    def reverse_verbose(self, x):
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)

        if self.u_proj is not None:
            x = x @ self.u_proj.t()

        outs = [x]

        x1, x2 = torch.chunk(x, 2, dim=-1)

        for layer in reversed(self.layers):
            x1, x2 = layer.reverse(x1, x2)
            outs.append(torch.cat((x1, x2), dim=-1))

        outs.reverse()
        return outs
