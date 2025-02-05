import torch
from torch import nn as nn
import torch.autograd.forward_ad as fwAD

from fastbreak import (
    to_ntuple,
    ReversibleBlock,
    FastBreakLinearBase,
    ReversibleSequence,
)


__all__ = ["RevMLP"]


class RevMLP(nn.Module):
    def __init__(
        self,
        feature_dim,
        num_layers,
        loss_func,
        hidden_dim=None,
        output_dim=None,
        layer_norm=False,
        bias=False,
        non_linearity=None,
        inv_method="default",
        inv_method_args=None,
        split_dim=-1,
        gain=1.0,
        proj_gain=1.0,
        train_ggn=True,
    ):
        super(RevMLP, self).__init__()

        self.feature_dim = feature_dim
        self.split_dim = split_dim

        self.loss_func = loss_func
        self.train_ggn = train_ggn

        to_n_layer_tuple = to_ntuple(num_layers)

        if non_linearity == True:
            non_linearity = nn.ReLU()
        if not isinstance(non_linearity, list):
            if isinstance(non_linearity, str):
                non_linearity = getattr(nn, non_linearity)()
            self.non_linearity = to_n_layer_tuple(non_linearity)
        else:
            assert len(non_linearity) == num_layers
            self.non_linearity = non_linearity

        self.num_layers = num_layers

        self.layers = [
            ReversibleBlock(
                FastBreakLinearBase(
                    feature_dim // 2,
                    feature_dim // 2,
                    hidden_dim=hidden_dim,
                    non_linearity=nn.Identity(),
                    norm_layer=None,
                    bias=bias,
                    gain=gain,
                ),
                FastBreakLinearBase(
                    feature_dim // 2,
                    feature_dim // 2,
                    hidden_dim=hidden_dim,
                    non_linearity=nn.Identity(),
                    norm_layer=None,
                    bias=bias,
                    gain=gain,
                ),
                inv_method=inv_method,
                inv_method_kwargs=inv_method_args,
                split_along_dim=split_dim,
            ),
        ]
        for n in range(1, num_layers):
            self.layers.append(
                ReversibleBlock(
                    FastBreakLinearBase(
                        feature_dim // 2,
                        feature_dim // 2,
                        hidden_dim=hidden_dim,
                        non_linearity=self.non_linearity[n],
                        norm_layer=None,
                        bias=bias,
                        gain=gain,
                    ),
                    FastBreakLinearBase(
                        feature_dim // 2,
                        feature_dim // 2,
                        hidden_dim=hidden_dim,
                        non_linearity=self.non_linearity[n],
                        norm_layer=None,
                        bias=bias,
                        gain=gain,
                    ),
                    split_along_dim=split_dim,
                    inv_method=inv_method,
                    inv_method_kwargs=inv_method_args,
                )
            )
        self.num_layers = len(self.layers)
        self.layers = ReversibleSequence(
            nn.ModuleList(self.layers),
        )

        if output_dim:
            u, _ = torch.linalg.qr(torch.randn(feature_dim, output_dim))
            self.register_buffer("u_proj", u)
        else:
            self.u_proj = None

    def forward(self, x, record_gradients=False):
        x = x.flatten(1)
        return self.layers(
            x, record_gradients=record_gradients, train_ggn=self.train_ggn
        )

    def reverse(self, out):
        if self.u_proj is not None:
            out = out @ self.u_proj.t()

        out = self.layers.reverse(out)

        return out

    def backward(self, out, label, logger=None, layer_n=None):
        """We always save outputs and targets in the same 'dimension order' as the input (usually, (batch_size, num_channels, feature_dim)).
        Similarly, we expect 'out' to be of same dimension order as the input."""
        pseudo_inv_hessian_epsilon = self.loss_func.compute_pseudoinv_hessian_epsilon(
            output=out,
            target=label,
            U_proj=self.u_proj,
        )

        with fwAD.dual_level():
            x_prev = fwAD.make_dual(tensor=out, tangent=pseudo_inv_hessian_epsilon)

            x_prev = self.layers.backward(
                x_prev,
            )

        return x_prev
