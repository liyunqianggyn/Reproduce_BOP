
from typing import TypeVar, Union, Tuple, Optional, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as f
import os
from torch import Tensor
from torch.autograd import Function
from torch.optim.optimizer import Optimizer
from torch.optim import Adam, SGD
import numpy as np

################################################################################

T = TypeVar("T")
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_size_2_t = _scalar_or_tuple_2_t[int]

################################################################################
# Quantizers

def l2_penalty(w):
    return w

class Binarize(Function):
    clip_value = 1

    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)

        output = inp.sign()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp: Tensor = ctx.saved_tensors[0]

        clipped = inp.abs() <= Binarize.clip_value

        output = torch.zeros(inp.size()).to(grad_output.device)
        output[clipped] = 1
        output[~clipped] = 0

        return output * grad_output


binarize = Binarize.apply

################################################################################

class Bopoptimizer(Optimizer):
    def __init__(
        self,
        binary_params,
        ar: float = 0.0001,
        threshold: float = 0,
        adam_lr=0.001,
    ):
        if not -1 <= ar:
            raise ValueError(
                "given adaptivity rate {} is invalid; should be in (0, 1) (excluding endpoints)".format(
                    ar
                )
            )
        if threshold < 0:
            raise ValueError(
                "given threshold {} is invalid; should be > 0".format(threshold)
            )

        self.total_weights = {}

        defaults = dict(adaptivity_rate=ar, threshold=threshold)
        super(Bopoptimizer, self).__init__(
            binary_params, defaults
        )

    def step(self, closure: Optional[Callable[[], float]] = ..., ar=None):

        flips = {None}

        for group in self.param_groups:
            params = group["params"]

            y = group["adaptivity_rate"]
            t = group["threshold"]
            flips = {}

            if ar is not None:
                y = ar

            for param_idx, p in enumerate(params):
                grad = p.grad.data
                state = self.state[p]

                if "moving_average" not in state:
                    m = state["moving_average"] = torch.clone(grad).detach()
                else:
                    m: Tensor = state["moving_average"]

                    m.mul_((y))
                    m.add_(grad.mul(1-y))


                mask = (m.abs() >= t) * (m.sign() == p.sign())
                mask = mask.float() * -1
                mask[mask == 0] = 1

                flips[param_idx] = (mask == -1).sum().item()

                p.data.mul_(mask)

        return flips

    def zero_grad(self) -> None:
        super().zero_grad()
        



