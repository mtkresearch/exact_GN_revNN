import torch
import torch.autograd.forward_ad as fwAD
import numpy as np
import math
from copy import deepcopy

from ..layers import FastBreakLinearBase
from ..utils import compute_ggn_update

__ALL__ = ["ReversibleLinear"]


class ReversibleLinear(FastBreakLinearBase):
    def __init__(
        self,
        feature_dim,
        hidden_dim=None,
        non_linearity=None,
        first_layer=False,
        layer_norm=None,
        bias=False,
        transpose_layer=False,
        inv_method="default",
        inv_method_args=None,
        rtol=0.0,
        atol=1e-7,
        split_dim=-1,
        transp_layers_norm_dim=None,
        gain=1.0,
        proj_gain=1.0,
        debug_mode=False,
    ):
        super().__init__(
            feature_dim,
            feature_dim,
            hidden_dim=hidden_dim,
            non_linearity=non_linearity,
            bias=bias,
            first_layer=first_layer,
            norm_layer=layer_norm,
            transpose_layer=transpose_layer,
            transp_layers_norm_dim=transp_layers_norm_dim,
            gain=gain,
            proj_gain=proj_gain,
        )

        self.split_dim = split_dim

        self.inv_method = inv_method
        self.inv_method_args = deepcopy(inv_method_args)

        self.rtol = rtol
        self.atol = atol

        self.debug_mode = debug_mode

    def forward(self, x1, x2):
        return x2, super().forward(x2) + x1

    def reverse(self, y1, y2):
        return y2 - super().forward(y1), y1

    def backward(self, x, logger=None, layer_n=None) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=self.split_dim)
        x1_prev, x2_prev = self.reverse(x1, x2)
        jvp_dgdz_hepsilon = fwAD.unpack_dual(x2).tangent

        if self.debug_mode:
            self.jvp_dgdz_hepsilon = jvp_dgdz_hepsilon

        with torch.no_grad():
            a = self._pre_forward(fwAD.unpack_dual(x2_prev)[0])
            a = a.flatten(0, -2)

            if self.bias is not None:
                one = torch.ones(a.shape[0], 1, device=a.device, dtype=a.dtype)
                a = torch.cat((a, one), dim=1)

            batch_size, d = x2_prev.shape[0], x2_prev.shape[1]
            jvp_dgdz_hepsilon = jvp_dgdz_hepsilon.flatten(0, -2)

            ggn_update = compute_ggn_update(
                input=a,
                jvp=jvp_dgdz_hepsilon,
                batch_size=batch_size,
                d=d,
                method=self.inv_method,
                method_kwargs=self.inv_method_args,
            )

            x_prev = torch.cat((x1_prev, x2_prev), dim=self.split_dim)

            if self.bias is not None:
                ggn_update_weight, ggn_update_bias = torch.split(
                    ggn_update, [self.weight.shape[1], 1], dim=0
                )
                self.weight.grad = ggn_update_weight.transpose(0, 1)
                self.bias.grad = ggn_update_bias.squeeze()
            else:
                self.weight.grad = ggn_update.transpose(0, 1)

        return x_prev

    def log_results(
        self,
        logger,
        layer_n,
        x,
        S,
        jvp_dgdz_hepsilon,
        out_rev_plus,
        error,
    ) -> None:
        # logger.logkv(
        #     f"layer {layer_n} output activation mean",
        #     x.mean(),
        # )
        # logger.logkv(
        #     f"layer {layer_n} output activation std",
        #     x.std(),
        # )
        # logger.logkv(
        #     f"layer {layer_n} Sigma Condition number",
        #     S.max() / S.min(),
        # )
        logger.logkv(
            f"layer {layer_n} output activation mean",
            x.mean(),
        )
        logger.logkv(
            f"layer {layer_n} output activation std",
            x.std(),
        )

        if S is not None:
            logger.logkv(
                f"layer {layer_n} Sigma Condition number",
                S.max() / S.min(),
            )
            logger.logkv(
                f"layer {layer_n} Sigma Max Singular Value",
                S.max(),
            )
            logger.logkv(
                f"layer {layer_n} Sigma Min Singular Value",
                S.min(),
            )
            logger.logkv(
                f"layer {layer_n} Sigma Median Singular Value",
                S.median(),
            )

        logger.logkv(
            f"layer {layer_n} Weight Norm",
            torch.norm(self.weight, p=2) / math.sqrt(self.weight.numel()),
        )

        # logger.logkv(
        #     f"layer {layer_n} Sigma Median Singular Value",
        #     S.median(),
        # )
        # logger.logkv(
        #     f"layer {layer_n} JVP Norm",
        #     torch.norm(jvp_dgdz_hepsilon),
        # )
        # logger.logkv(
        #     f"layer {layer_n} Sigma-plus Rank",
        #     torch.linalg.matrix_rank(out_rev_plus).item(),
        # )

        logger.logkv(
            f"layer {layer_n} Update Norm",
            (
                torch.norm(self.weight.grad, p=2) / math.sqrt(self.weight.grad.numel())
                if self.weight.grad is not None
                else 0.0
            ),
        )

        if error is not None:
            logger.logkv(f"layer {layer_n} approx error", error)
