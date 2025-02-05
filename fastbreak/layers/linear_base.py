import torch
from torch import nn
from torch import Tensor
from functools import partial

from typing import Callable, Any, TYPE_CHECKING, Tuple, List

__all__ = [
    "FastBreakLinearBase",
]


class FastBreakLinearBase(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int,
        non_linearity: Callable,  # TODO: Be more specific here??
        bias: bool = True,
        device: Any or None = None,
        dtype: Any or None = None,
        first_layer: bool = False,
        norm_layer: partial
        or None = partial(
            torch.nn.LayerNorm, eps=1e-6, elementwise_affine=False  # , bias=False
        ),
        transpose_layer=False,
        transp_layers_norm_dim=None,
        gain: float = 1.0,
        proj_gain: float = 1.0,
    ) -> None:
        super().__init__(
            in_features=hidden_dim if hidden_dim else in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self.first_layer = first_layer
        self.non_linearity = non_linearity
        self.init_parameters(gain=gain)

        if hidden_dim:
            sigma = proj_gain / (hidden_dim**0.5)

            self.C = nn.Parameter(
                torch.randn(in_features, hidden_dim) * sigma, requires_grad=False
            ).to(device)
        else:
            self.C = None

        if norm_layer is not None and not isinstance(norm_layer, nn.Identity):
            self.norm_layer = norm_layer(
                normalized_shape=(
                    in_features if not transpose_layer else transp_layers_norm_dim
                ),
                device=device,
                dtype=dtype,
            )
        else:
            self.norm_layer = None

        self.transpose_layer = transpose_layer

    # weight initialization
    def init_parameters(self, gain):
        if isinstance(self.non_linearity, nn.Identity):
            default_gain = 1.0
        elif isinstance(self.non_linearity, nn.ReLU):
            default_gain = nn.init.calculate_gain("relu")
        else:
            raise NotImplementedError(
                f"Non-linearity {self.non_linearity} not supported."
            )

        nn.init.xavier_normal_(self.weight, gain=gain * default_gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _pre_forward(
        self,
        input: torch.Tensor,
    ) -> Tuple[Tensor, Tensor or Any]:
        pre_out = input

        if self.norm_layer is not None:
            if self.transpose_layer:
                pre_out = pre_out.transpose(1, 2)

            pre_out = self.norm_layer(pre_out)

            if self.transpose_layer:
                pre_out = pre_out.transpose(1, 2)

        if self.C is not None:
            pre_out = pre_out @ self.C

        if (
            not self.first_layer and self.non_linearity
        ):  # usually don't want to apply non-linearity directly to the input
            pre_out = self.non_linearity(pre_out)

        return pre_out

    def _forward(
        self,
        input: torch.Tensor,
        skip_input: torch.Tensor or List[torch.Tensor] or None = None,
    ) -> Tuple[Tensor, Tensor or Any]:
        out = super().forward(input)

        if skip_input is not None:
            if isinstance(skip_input, list):
                out = out + sum(skip_input)
            else:
                out = out + skip_input
        return out

    def forward(
        self,
        input: torch.Tensor,
        skip_input: None or torch.Tensor or List[torch.Tensor] = None,
    ) -> Tensor:
        pre_out = self._pre_forward(input)

        return self._forward(pre_out, skip_input=skip_input)

    def update_grad(self, update):
        if self.bias is not None:
            ggn_update_weight, ggn_update_bias = torch.split(
                update, [self.weight.shape[1], 1], dim=0
            )
            self.weight.grad = ggn_update_weight.transpose(0, 1)
            self.bias.grad = ggn_update_bias.squeeze()
        else:
            self.weight.grad = update.transpose(0, 1)
