import random
import sys

import torch
import torch.nn as nn
import torch.autograd.forward_ad as fwAD

from torch.autograd import Function

from fastbreak.utils import compute_pinverse

__ALL__ = ["ReversibleBlock"]


class ReversibleBlock(nn.Module):
    """
    Elementary building block for building (partially) reversible architectures

    Implementation of the Reversible block described in the RevNet paper
    (https://arxiv.org/abs/1707.04585). Must be used inside a :class:`revtorch.ReversibleSequence`
    for autograd support.

    Arguments:
        f_block (nn.Module): arbitrary subnetwork whos output shape is equal to its input shape
        g_block (nn.Module): arbitrary subnetwork whos output shape is equal to its input shape
        split_along_dim (integer): dimension along which the tensor is split into the two parts requried for the reversible block
        fix_random_seed (boolean): Use the same random seed for the forward and backward pass if set to true
    """

    def __init__(
        self,
        f_block: nn.Module,
        g_block: nn.Module,
        split_along_dim: int = 1,
        fix_random_seed: bool = False,
        inv_method: str = "default",
        inv_method_kwargs: dict = {},
    ):
        super(ReversibleBlock, self).__init__()
        self.f_block = f_block
        self.g_block = g_block
        self.split_along_dim = split_along_dim
        self.fix_random_seed = fix_random_seed
        self.random_seeds = {}
        self.inv_method = inv_method
        self.inv_method_kwargs = inv_method_kwargs

    def _init_seed(self, namespace):
        if self.fix_random_seed:
            self.random_seeds[namespace] = random.randint(0, sys.maxsize)
            self._set_seed(namespace)

    def _set_seed(self, namespace):
        if self.fix_random_seed:
            torch.manual_seed(self.random_seeds[namespace])

    def forward(self, x, record_gradients=True):
        """
        Performs the forward pass of the reversible block. Does not record any gradients.
        :param x: Input tensor. Must be splittable along dimension 1.
        :return: Output tensor of the same shape as the input tensor
        """
        x1, x2 = torch.chunk(x, 2, dim=self.split_along_dim)
        y1, y2 = None, None
        if record_gradients:
            self._init_seed("f")
            y1 = x1 + self.f_block(x2)
            self._init_seed("g")
            y2 = x2 + self.g_block(y1)
        else:
            with torch.no_grad():
                self._init_seed("f")
                y1 = x1 + self.f_block(x2)
                self._init_seed("g")
                y2 = x2 + self.g_block(y1)

        return torch.cat([y1, y2], dim=self.split_along_dim)

    def pre_forward(self, x, exit_block):
        x1, x2 = torch.chunk(x, 2, dim=self.split_along_dim)
        if exit_block == "f_block":
            self._init_seed("f")
            return self.f_block._pre_forward(x2)
        else:
            self._init_seed("f")
            y1 = x1 + self.f_block(x2)
            self._init_seed("g")
            return self.g_block._pre_forward(y1)

    def forward_with_weight(self, x, weight):
        x1, x2 = torch.chunk(x, 2, dim=self.split_along_dim)
        y1, y2 = None, None
        self._init_seed("f")

        expected_shape_with_bias = (
            self.f_block.weight.shape[0],
            self.f_block.weight.shape[1] + 1,
        )

        if weight.shape == expected_shape_with_bias:
            f_weight = weight[:, :-1]
            f_bias = weight[:, -1]
            y1 = x1 + torch.matmul(self.f_block._pre_forward(x2), f_weight.t()) + f_bias
        elif weight.shape == self.f_block.weight.shape:
            f_weight = weight
            y1 = x1 + torch.matmul(self.f_block._pre_forward(x2), f_weight.t())
        else:
            NotImplementedError(
                f"The shape of weight is {weight.shape}, but expected shape is {expected_shape_with_bias} or {self.f_block.weight.shape}."
            )

        self._init_seed("g")
        y2 = x2 + self.g_block(y1)
        return y2

    def reverse(self, y):
        """
        Performs the reverse pass of the reversible block.

        :param y: Outputs of the reversible block
        :return: A tensor of block inputs. The block inputs are the same shape as the block outptus.
        """

        # Split the arguments channel-wise
        y1, y2 = torch.chunk(y, 2, dim=self.split_along_dim)
        del y

        self._set_seed("g")
        gy1 = self.g_block(y1)

        x2 = y2 - gy1  # Restore first input of forward()
        del y2, gy1

        self._set_seed("f")
        fx2 = self.f_block(x2)

        x1 = y1 - fx2  # Restore second input of forward()
        del y1, fx2

        # Undo the channelwise split
        x = torch.cat([x1, x2], dim=self.split_along_dim)

        return x

    def backward(self, y):
        x_prev = self.reverse(y)
        jvp_dgdz_hepsilon = fwAD.unpack_dual(y).tangent
        jvp1, jvp2 = torch.chunk(jvp_dgdz_hepsilon, 2, dim=self.split_along_dim)

        # update for F
        a_f = self.pre_forward(x_prev, exit_block="f_block")
        a_f = a_f.flatten(0, -2)

        if self.f_block.bias is not None:
            one = torch.ones(a_f.shape[0], 1, device=a_f.device, dtype=a_f.dtype)
            a_f = torch.cat((a_f, one), dim=1)

        a_f_plus = compute_pinverse(a_f, self.inv_method, self.inv_method_kwargs)
        update_F = a_f_plus @ jvp1

        a_g = self.pre_forward(x_prev, exit_block="g_block")
        a_g = a_g.flatten(0, -2)

        if self.g_block.bias is not None:
            one = torch.ones(a_g.shape[0], 1, device=a_g.device, dtype=a_g.dtype)
            a_g = torch.cat((a_g, one), dim=1)

        a_g_plus = compute_pinverse(a_g, self.inv_method, self.inv_method_kwargs)

        # contruct the vector for the jvp
        jvp_vector = update_F.clone().detach()

        # assign vector as tanget to weight (and bias)
        # nn.Parameter cannot be used for dual, so need to replace it and then put it back
        jvp_vector_weight = jvp_vector
        if self.f_block.bias is not None:
            jvp_vector_weight, jvp_vector_bias = torch.split(
                jvp_vector, [self.f_block.weight.shape[1], 1], dim=0
            )
            jvp_vector_bias = jvp_vector_bias.squeeze()

        w_f = self.f_block.weight.data
        if self.f_block.bias is not None:
            w_b = self.f_block.bias.data

        # compute output so we get the jvp
        w_dual = fwAD.make_dual(tensor=w_f, tangent=jvp_vector_weight.detach().T)
        if self.f_block.bias is not None:
            w_b_dual = fwAD.make_dual(tensor=w_b, tangent=jvp_vector_bias.detach())
            w_dual = torch.cat((w_dual, w_b_dual.unsqueeze(1)), dim=1)

        out = self.forward_with_weight(fwAD.unpack_dual(x_prev)[0].detach(), w_dual)
        missing_jacobian = fwAD.unpack_dual(out).tangent
        # if self.g_block.bias is not None:
        # missing_jacobian_bias = fwAD.unpack_dual

        # reassign parameters (also need to reassign to optimizer)
        # self.g_block.weight = nn.Parameter(
        #     fwAD.unpack_dual(self.g_block.weight)[0], requires_grad=True
        # )
        # if self.g_block.bias is not None:
        #     self.g_block.bias = nn.Parameter(
        #         fwAD.unpack_dual(self.g_block.bias)[0], requires_grad=True
        #     )

        # finish computing update for G
        full_missing_component = a_g_plus @ missing_jacobian
        first_half = a_g_plus @ jvp2
        update_G = first_half - full_missing_component

        # put update in grad attribute
        self.f_block.update_grad(update_F)
        self.g_block.update_grad(update_G)

        return x_prev

    def backward_pass(self, y, dy, retain_graph):
        """
        Performs the backward pass of the reversible block.

        Calculates the derivatives of the block's parameters in f_block and g_block, as well as the inputs of the
        forward pass and its gradients.

        :param y: Outputs of the reversible block
        :param dy: Derivatives of the outputs
        :param retain_graph: Whether to retain the graph on intercepted backwards
        :return: A tuple of (block input, block input derivatives). The block inputs are the same shape as the block outptus.
        """

        # Split the arguments channel-wise
        y1, y2 = torch.chunk(y, 2, dim=self.split_along_dim)
        del y
        assert not y1.requires_grad, "y1 must already be detached"
        assert not y2.requires_grad, "y2 must already be detached"
        dy1, dy2 = torch.chunk(dy, 2, dim=self.split_along_dim)
        del dy
        assert not dy1.requires_grad, "dy1 must not require grad"
        assert not dy2.requires_grad, "dy2 must not require grad"

        # Enable autograd for y1 and y2. This ensures that PyTorch
        # keeps track of ops. that use y1 and y2 as inputs in a DAG
        y1.requires_grad = True
        y2.requires_grad = True

        # Ensures that PyTorch tracks the operations in a DAG
        with torch.enable_grad():
            self._set_seed("g")
            gy1 = self.g_block(y1)

            # Use autograd framework to differentiate the calculation. The
            # derivatives of the parameters of G are set as a side effect
            gy1.backward(dy2, retain_graph=retain_graph)

        with torch.no_grad():
            x2 = y2 - gy1  # Restore first input of forward()
            del y2, gy1

            # The gradient of x1 is the sum of the gradient of the output
            # y1 as well as the gradient that flows back through G
            # (The gradient that flows back through G is stored in y1.grad)
            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            self._set_seed("f")
            fx2 = self.f_block(x2)

            # Use autograd framework to differentiate the calculation. The
            # derivatives of the parameters of F are set as a side effec
            fx2.backward(dx1, retain_graph=retain_graph)

        with torch.no_grad():
            x1 = y1 - fx2  # Restore second input of forward()
            del y1, fx2

            # The gradient of x2 is the sum of the gradient of the output
            # y2 as well as the gradient that flows back through F
            # (The gradient that flows back through F is stored in x2.grad)
            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            # Undo the channelwise split
            x = torch.cat([x1, x2.detach()], dim=self.split_along_dim)
            dx = torch.cat([dx1, dx2], dim=self.split_along_dim)

        return x, dx


class _ReversibleModuleFunction(torch.autograd.function.Function):
    """
    Integrates the reversible sequence into the autograd framework
    """

    @staticmethod
    def forward(ctx, x, reversible_blocks, eagerly_discard_variables):
        """
        Performs the forward pass of a reversible sequence within the autograd framework
        :param ctx: autograd context
        :param x: input tensor
        :param reversible_blocks: nn.Modulelist of reversible blocks
        :return: output tensor
        """
        assert isinstance(reversible_blocks, nn.ModuleList)
        for block in reversible_blocks:
            assert isinstance(block, ReversibleBlock)
            x = block(x)
        ctx.y = (
            x.detach()
        )  # not using ctx.save_for_backward(x) saves us memory by beeing able to free ctx.y earlier in the backward pass
        ctx.reversible_blocks = reversible_blocks
        ctx.eagerly_discard_variables = eagerly_discard_variables
        return x

    @staticmethod
    def backward(ctx, dy):
        """
        Performs the backward pass of a reversible sequence within the autograd framework
        :param ctx: autograd context
        :param dy: derivatives of the outputs
        :return: derivatives of the inputs
        """
        y = ctx.y
        if ctx.eagerly_discard_variables:
            del ctx.y
        for i in range(len(ctx.reversible_blocks) - 1, -1, -1):
            y, dy = ctx.reversible_blocks[i].backward_pass(
                y, dy, not ctx.eagerly_discard_variables
            )
        if ctx.eagerly_discard_variables:
            del ctx.reversible_blocks
        return dy, None, None


class ReversibleSequence(nn.Module):
    """
    Basic building element for (partially) reversible networks

    A reversible sequence is a sequence of arbitrarly many reversible blocks. The entire sequence is reversible.
    The activations are only saved at the end of the sequence. Backpropagation leverages the reversible nature of
    the reversible sequece to save memory.

    Arguments:
        reversible_blocks (nn.ModuleList): A ModuleList that exclusivly contains instances of ReversibleBlock
        which are to be used in the reversible sequence.
        eagerly_discard_variables (bool): If set to true backward() discards the variables requried for
                calculating the gradient and therefore saves memory. Disable if you call backward() multiple times.
    """

    def __init__(self, reversible_blocks, eagerly_discard_variables=True):
        super(ReversibleSequence, self).__init__()
        assert isinstance(reversible_blocks, nn.ModuleList)
        for block in reversible_blocks:
            assert isinstance(block, ReversibleBlock)

        self.reversible_blocks = reversible_blocks
        self.eagerly_discard_variables = eagerly_discard_variables

    def forward(self, x, record_gradients=False, train_ggn=False):
        """
        Forward pass of a reversible sequence
        :param x: Input tensor
        :return: Output tensor
        """
        if train_ggn:
            for i, block in enumerate(self.reversible_blocks):
                x = block(x, record_gradients=record_gradients)
        else:
            x = _ReversibleModuleFunction.apply(
                x, self.reversible_blocks, self.eagerly_discard_variables
            )
        return x

    def backward(self, x):
        for i, block in enumerate(reversed(self.reversible_blocks)):
            x = block.backward(x)
        return x

    def reverse(self, x):
        for i, block in enumerate(reversed(self.reversible_blocks)):
            x = block.reverse(x)
        return x
