import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import torch.autograd.forward_ad as fwAD
from functools import partial

from .linear_base import FastBreakLinearBase

from einops import rearrange

from typing import Callable, Any, TYPE_CHECKING, Tuple, List

if TYPE_CHECKING:
    from .. import FastBreakLossFunction

__all__ = [
    "FastBreakLinear",
]


class FastBreakLinear(FastBreakLinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int,
        loss_fn: "FastBreakLossFunction",
        non_linearity: Callable,
        bias: bool = True,
        device: Any or None = None,
        dtype: Any or None = None,
        first_layer: bool = False,
        norm_layer: partial
        or None = partial(
            torch.nn.LayerNorm, eps=1e-6, elementwise_affine=False  # , bias=False
        ),
        pooling: str or None = None,
        transpose_layer: bool = False,
        transp_layers_norm_dim=None,
        gain: float = 1.0,
        proj_gain: float = 1.0,
        alpha: Tensor = None,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_dim=hidden_dim,
            non_linearity=non_linearity,
            bias=bias,
            device=device,
            dtype=dtype,
            first_layer=first_layer,
            norm_layer=norm_layer,
            pooling=pooling,
            transpose_layer=transpose_layer,
            transp_layers_norm_dim=transp_layers_norm_dim,
            gain=gain,
            proj_gain=proj_gain,
        )

        self.loss_fn = loss_fn
        self.alpha = alpha

        self.weight_handle = None
        self.bias_handle = None

    def forward(
        self,
        input: torch.Tensor,
        target: None or torch.Tensor = None,
        target_prev: None or torch.Tensor = None,
        skip_input: None or torch.Tensor or List[torch.Tensor] = None,
        skip_target: None or torch.Tensor or List[torch.Tensor] = None,
        U_prev: None or torch.Tensor = None,
        U_skip: None or torch.Tensor or List[torch.Tensor] = None,
        alpha_prev: None or torch.Tensor = None,
        alpha_skip: None or torch.Tensor = None,
    ) -> Tensor:
        if (
            self.training and self.weight.requires_grad
        ):  # compute GGN update during forward pass
            assert target is not None

            if self.weight_handle:
                self.weight_handle.remove()
            if self.bias_handle:
                self.bias_handle.remove()

            with fwAD.dual_level():
                if not self.first_layer:
                    assert target_prev is not None
                    pseudo_inv_hessian_epsilon_prev = (
                        self.loss_fn.compute_pseudoinv_hessian_epsilon(
                            output=input, target=target_prev, U_proj=U_prev
                        )
                    )
                    layer_input = fwAD.make_dual(
                        tensor=input, tangent=pseudo_inv_hessian_epsilon_prev
                    )
                else:
                    layer_input = input

                final_output, output_pre_w = super().forward(
                    layer_input, skip_input=skip_input
                )

                with torch.no_grad():
                    pseudo_inv_hessian_epsilon = (
                        self.loss_fn.compute_pseudoinv_hessian_epsilon(
                            output=final_output, target=target, U_proj=self.U_proj
                        )
                    )

                    if self.alpha is None:
                        self.alpha = 1.0
                    if alpha_prev is None:
                        alpha_prev = 1.0
                    if alpha_skip is None:
                        alpha_skip = 1.0

                    if self.first_layer:
                        # in first layer G = Z_1 - W\Sigma(X)
                        jvp_dgdz_hepsilon = self.alpha * pseudo_inv_hessian_epsilon
                    else:
                        jvp_dg_dz_prev = fwAD.unpack_dual(
                            final_output
                        ).tangent  # JVP for z_{l-1}
                        jvp_dgdz_hepsilon = (
                            self.alpha * pseudo_inv_hessian_epsilon
                        ) - (alpha_prev * jvp_dg_dz_prev)

                        if skip_target is not None:
                            if isinstance(skip_target, list):
                                if U_skip is None:
                                    U_skip = [None for _ in range(len(skip_target))]
                                for si, st, usi in zip(skip_input, skip_target, U_skip):
                                    jvp_dgdz_hepsilon -= alpha_skip * (
                                        self.loss_fn.compute_pseudoinv_hessian_epsilon(
                                            output=si, target=st, U_proj=usi
                                        )
                                    )
                            else:
                                jvp_dgdz_hepsilon -= alpha_skip * (
                                    self.loss_fn.compute_pseudoinv_hessian_epsilon(
                                        output=skip_input,
                                        target=skip_target,
                                        U_proj=U_skip,
                                    )
                                )

                    batch_size = input.shape[0]
                    if output_pre_w.dim() == 3:
                        num_patches = output_pre_w.shape[1]
                        output_pre_w = output_pre_w.reshape(
                            batch_size * num_patches, -1
                        )
                        one = torch.ones(
                            batch_size * num_patches, 1, device=input.device
                        )

                        jvp_dgdz_hepsilon = jvp_dgdz_hepsilon.reshape(
                            batch_size * num_patches, -1
                        )
                    else:
                        assert output_pre_w.dim() == 2
                        one = torch.ones(batch_size, 1, device=input.device)

                    # Pseudo-invert Sigma
                    one = torch.ones(output_pre_w.shape[0], 1, device=input.device)

                    sigma = torch.cat((output_pre_w, one), dim=1)

                    pinv_sigma = torch.pinverse(sigma)

                    ggn_update = pinv_sigma @ jvp_dgdz_hepsilon
                    ggn_update_weight, ggn_update_bias = torch.split(
                        ggn_update, [self.weight.shape[1], 1], dim=0
                    )

            def backward_hook(grad) -> Tensor:
                grad_out = ggn_update_weight.transpose(0, 1)
                return grad_out

            def backward_hook_bias(grad) -> Tensor:
                grad_out = ggn_update_bias.squeeze().detach()
                return grad_out

            self.weight_handle = self.weight.register_hook(backward_hook)
            self.bias_handle = self.bias.register_hook(backward_hook_bias)
            return final_output
        else:
            final_output, _ = super().forward(input, skip_input=skip_input)
            return final_output
