import numpy as np
import sys
import torch
import torch.nn.functional as F
import math
from kan import KANLinear
import convolution
class KAN_Convolution_Linears(torch.nn.Module):
    def __init__(
        self,
        in_features = (28,28),
        kernel_size= (2,2),
        stride = 1,
        padding=None,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=False,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN_Convolution_Linears, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.layers = torch.nn.ModuleList()
        for i in range(math.prod(kernel_size)):
            self.layers.append(
                KANLinear(
                    in_features = 1,
                    out_features = 1,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
        

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
