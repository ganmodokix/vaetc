from turtle import forward
from typing import Any
import torch
from torch import nn

class RescaleGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_forward: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(scale, device=input_forward.device))
        return input_forward.view_as(input_forward)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        scale, = ctx.saved_tensors
        return grad_output * scale, None

def reverse_gradient(x: torch.Tensor) -> torch.Tensor:
    return RescaleGradient.apply(x, -1)

def rescale_gradient(x: torch.Tensor, scale: float) -> torch.Tensor:
    return RescaleGradient.apply(x, scale)

class GradientRescale(nn.Module):

    def __init__(self, scale: float) -> None:
        super().__init__()

        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return RescaleGradient.apply(x, self.scale)

class GradientReversal(GradientRescale):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return RescaleGradient.apply(x, -1)
