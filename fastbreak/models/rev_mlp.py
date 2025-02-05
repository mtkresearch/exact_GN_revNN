import torch
import torch.nn as nn
import torch.autograd.forward_ad as fwAD
from functools import partial

from .rev_linear import ReversibleLinear
from ..utils import to_ntuple

__ALL__ = ["ReversibleMLP"]


class ReversibleMLP(nn.Module):
    def __init__(
        self,
        feature_dim,
        num_layers,
        hidden_dim=None,
        output_dim=None,
        bias=False,
        transpose_layer=False,
        layer_norm=None,
        non_linearity=None,
        first_layer=True,
        inv_method="default",
        inv_method_args=None,
        rtol=0.0,
        atol=1e-7,
        split_dim=-1,
        transp_layers_norm_dim=None,
        gain=1.0,
        proj_gain=1.0,
    ):
        super(ReversibleMLP, self).__init__()

        self.feature_dim = feature_dim
        self.split_dim = split_dim

        self.transpose_layer = transpose_layer

        to_n_layer_tuple = to_ntuple(num_layers)

        if not isinstance(non_linearity, list):
            self.non_linearity = to_n_layer_tuple(non_linearity)
        else:
            assert len(non_linearity) == num_layers
            self.non_linearity = non_linearity

        if not isinstance(layer_norm, list):
            self.layer_norm = to_n_layer_tuple(layer_norm)
        else:
            assert len(layer_norm) == num_layers
            self.layer_norm = layer_norm

        self.num_layers = num_layers
        self.layers = [
            ReversibleLinear(
                feature_dim // 2,
                first_layer=first_layer,
                layer_norm=None if first_layer else self.layer_norm[0],
                non_linearity=nn.Identity() if first_layer else self.non_linearity[0],
                hidden_dim=hidden_dim,
                bias=bias,
                inv_method=inv_method,
                inv_method_args=inv_method_args,
                rtol=rtol,
                atol=atol,
                split_dim=split_dim,
                transpose_layer=transpose_layer,
                transp_layers_norm_dim=transp_layers_norm_dim,
                gain=gain,
                proj_gain=proj_gain,
            ),
            ReversibleLinear(
                feature_dim // 2,
                first_layer=first_layer,
                layer_norm=None if first_layer else self.layer_norm[1],
                non_linearity=nn.Identity() if first_layer else self.non_linearity[1],
                hidden_dim=hidden_dim,
                bias=bias,
                inv_method=inv_method,
                inv_method_args=inv_method_args,
                rtol=rtol,
                atol=atol,
                split_dim=split_dim,
                transpose_layer=transpose_layer,
                transp_layers_norm_dim=transp_layers_norm_dim,
                gain=gain,
                proj_gain=proj_gain,
            ),
        ]
        for n in range(2, num_layers):
            self.layers.append(
                ReversibleLinear(
                    feature_dim // 2,
                    layer_norm=self.layer_norm[n],
                    non_linearity=self.non_linearity[n],
                    hidden_dim=hidden_dim,
                    bias=bias,
                    inv_method=inv_method,
                    inv_method_args=inv_method_args,
                    rtol=rtol,
                    atol=atol,
                    split_dim=split_dim,
                    transpose_layer=transpose_layer,
                    transp_layers_norm_dim=transp_layers_norm_dim,
                    gain=gain,
                    proj_gain=proj_gain,
                )
            )
        self.layers = nn.ModuleList(self.layers)

        if output_dim:
            u, _ = torch.linalg.qr(torch.randn(feature_dim, output_dim))
            self.u_proj = nn.Parameter(u, requires_grad=False)
        else:
            self.u_proj = None

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=self.split_dim)

        for layer in self.layers:
            x1, x2 = layer(x1, x2)

        x = torch.cat((x1, x2), dim=self.split_dim)

        return x

    def reverse(self, out):
        if self.u_proj is not None:
            out = out @ self.u_proj.t()

        x1, x2 = torch.chunk(out, 2, dim=self.split_dim)

        for layer in reversed(self.layers):
            x1, x2 = layer.reverse(x1, x2)

        x = torch.cat((x1, x2), dim=self.split_dim)

        return x

    def backward(self, x_prev, logger=None, layer_n=None):
        """We always save outputs and targets in the same 'dimension order' as the input (usually, (batch_size, num_channels, feature_dim)).
        Similarly, we expect 'out' to be of same dimension order as the input."""
        for n, layer in enumerate(reversed(self.layers)):
            x_prev = layer.backward(
                x_prev,
                logger,
                layer_n - n if layer_n is not None else len(self.layers) - n,
            )

        return x_prev

    def forward_verbose(self, x):
        outs = [x]

        x1, x2 = torch.chunk(x, 2, dim=-1)

        for layer in self.layers:
            x1, x2 = layer.forward(x1, x2)
            outs.append(torch.cat((x1, x2), dim=-1))

        return outs

    def reverse_verbose(self, x):
        if self.u_proj is not None:
            x = x @ self.u_proj.t()

        outs = [x]

        x1, x2 = torch.chunk(x, 2, dim=-1)

        for layer in reversed(self.layers):
            x1, x2 = layer.reverse(x1, x2)
            outs.append(torch.cat((x1, x2), dim=-1))

        outs.reverse()
        return outs
