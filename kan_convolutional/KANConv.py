import numpy as np
import sys
import torch
import torch.nn.functional as F
import math
from KANLinear import KANLinear
import convolution

#Script que contiene la implementación del kernel con funciones de activación.
class KAN_Convolutional_Layer(torch.nn.Module):
    def __init__(self,
            n_convs = 1,
            kernel_size= (2,2),
            stride = (1,1),
            padding=(0,0),
            dilation =  (1,1),
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
            device = "cuda"):
        super(KAN_Convolutional_Layer, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.device = device
        self.convs = torch.nn.ModuleList()
        self.n_convs = n_convs
        for i in range(n_convs):
            self.convs.append(
                KAN_Convolution(
                    kernel_size= kernel_size,
                    stride = stride,
                    padding=padding,
                    dilation = dilation,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    device = device))
        

    def forward(self, x: torch.Tensor, update_grid=False):

        return torch.stack([i.forward(x) for i in self.convs]).to(self.device).transpose(0,1)

class KAN_Convolution(torch.nn.Module):
    def __init__(
        self,
        kernel_size= (2,2),
        stride = (1,1),
        padding=(0,0),
        dilation = (1,1),
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        device = "cuda"):
        super(KAN_Convolution, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.device = device
        self.convs = KANLinear(
                    in_features = math.prod(kernel_size),
                    out_features = 1,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range)

    def forward(self, x: torch.Tensor, update_grid=False):
        return convolution.apply_filter_to_image(x, self.convs,self.kernel_size[0],self.padding,self.stride,self.dilation, rgb = False)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers)

