import torch
from torch import nn as nn
from torch.nn import functional as F
import torch.autograd.forward_ad as fwAD

from fastbreak import MSELoss


__all__ = ["InvertibleMLP"]


class Invertiblelayer_norm(nn.Module):
    def __init__(self, dim=-1, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, input):
        mean = torch.mean(input, dim=(self.dim,), keepdim=True)
        var = torch.square(input - mean).mean(dim=(self.dim,), keepdim=True)

        self.last_mean = mean
        self.last_denominator = torch.sqrt(var + self.eps)

        return (input - self.last_mean) / self.last_denominator

    def reverse(self, out):
        return (out * self.last_denominator) + self.last_mean


class InvertibleLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input):
        return torch.where(input > 0, input, input * self.negative_slope)

    def reverse(self, out):
        return torch.where(out > 0, out, out / self.negative_slope)


class InvertibleLinear(nn.Module):
    def __init__(
        self, in_features, out_features, layer_norm=False, non_linearity=False
    ):
        super().__init__()
        self.lin = nn.Linear(
            in_features=in_features, out_features=out_features, bias=False
        )
        self.layer_norm = Invertiblelayer_norm() if layer_norm else None
        self.non_linearity = InvertibleLeakyReLU() if non_linearity else None

    @property
    def weight(self):
        return self.lin.weight

    def forward(self, input):
        if self.layer_norm is not None:
            input = self.layer_norm(input)

        if self.non_linearity is not None:
            input = self.non_linearity(input)

        return self.lin(input)

    def reverse(self, out, logger=None):
        weight = self.lin.weight
        inv_weight = torch.pinverse(weight)
        input = F.linear(out, inv_weight)

        if self.non_linearity is not None:
            input = self.non_linearity.reverse(input)

        if self.layer_norm is not None:
            input = self.layer_norm.reverse(input)

        return input


class InvertibleMLP(nn.Module):
    def __init__(
        self,
        feature_dim,
        num_layers,
        layer_norm=False,
        non_linearity=False,
        loss_func=MSELoss(),
        hidden_dim=None,
        threshold=0.0,
    ):
        super(InvertibleMLP, self).__init__()

        self.feature_dim = feature_dim
        self.loss_func = loss_func

        self.layer_norm = layer_norm
        self.non_linearity = non_linearity

        self.num_layers = num_layers
        self.u_proj = None

        self.layers = nn.Sequential()

        for i in range(num_layers):
            layer = InvertibleLinear(
                feature_dim,
                feature_dim,
                self.layer_norm if i > 0 else False,
                self.non_linearity if i > 0 else False,
            )
            torch.nn.init.orthogonal_(layer.weight)
            self.layers.add_module(f"layer{i+1}", layer)

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)

        out = self.layers(x)

        return out

    def forward_verbose(self, x):
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)

        outs = [x]
        for layer in self.layers:
            x = layer.forward(x)
            outs.append(x)

        return outs

    def reverse(self, x):
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)

        for layer in reversed(self.layers):
            x = layer.reverse(x)

        return x

    def reverse_verbose(self, x):
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)

        outs = [x]
        for layer in reversed(self.layers):
            x = layer.reverse(x)
            outs.append(x)

        outs.reverse()
        return outs

    def backward(self, out, label, logger=None):
        """
        This should use the pseudo inverses of the layers weights in order to recalculate the input.
        """
        # out_rev = out
        pseudo_inv_hessian_epsilon = self.loss_func.compute_pseudoinv_hessian_epsilon(
            output=out, target=label
        )

        with fwAD.dual_level():
            out_rev = fwAD.make_dual(tensor=out, tangent=pseudo_inv_hessian_epsilon)
            jvp_dgdz_hepsilon_prev = pseudo_inv_hessian_epsilon
            for n, i in enumerate(reversed(range(0, self.num_layers))):
                layer = self.layers[i]

                out_rev = layer.reverse(out_rev)

                jvp_dgdz_hepsilon = fwAD.unpack_dual(out_rev).tangent

                U, S, Vt = torch.linalg.svd(out_rev, full_matrices=False)
                out_rev_plus = (
                    Vt.transpose(0, 1) @ torch.diag(1 / S) @ U.transpose(0, 1)
                )

                ggn_update = (
                    (1 / len(self.layers)) * out_rev_plus @ jvp_dgdz_hepsilon_prev
                )

                layer.weight.grad = ggn_update.transpose(0, 1)

                if logger:
                    logger.logkv(
                        f"layer {len(self.layers) - n} Sigma Condition number",
                        S.max() / S.min(),
                    )

                if logger:
                    logger.logkv(
                        f"layer {len(self.layers) - n} JVP Norm",
                        torch.norm(jvp_dgdz_hepsilon),
                    )
                    logger.logkv(
                        f"layer {len(self.layers) - n} Sigma-plus Rank",
                        torch.linalg.matrix_rank(out_rev_plus).item(),
                    )

                jvp_dgdz_hepsilon_prev = jvp_dgdz_hepsilon

                if logger:
                    logger.logkv(
                        f"layer {len(self.layers) - n} Update Grad",
                        (
                            torch.norm(layer.weight.grad)
                            if layer.weight.grad is not None
                            else 0.0
                        ),
                    )

        return out


if __name__ == "__main__":
    X = torch.randn((256, 784))
    label = torch.randn((256, 784))

    model = InvertibleMLP(in_channels=784, feature_dim=784, num_layers=1)

    out = model(X)

    X_in = model.reverse(out, label)

    assert torch.allclose(
        X_in, X, atol=1e-3, rtol=1e-3
    ), "Reconstructed input does not match the original input"

    print(next(iter(model.parameters())).grad)
