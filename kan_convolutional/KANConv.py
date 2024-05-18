import torch
import math
import sys
sys.path.append('./kan_convolutional')

import numpy as np
from KANLinear import KANLinear
import convolution
import sys
sys.path.append('./kan_convolutional')


#Script que contiene la implementación del kernel con funciones de activación.
class KAN_Convolutional_Layer(torch.nn.Module):
    def __init__(
            self,
            n_convs: int = 1,
            kernel_size: tuple = (2,2),
            stride: tuple = (1,1),
            padding: tuple = (0,0),
            dilation: tuple = (1,1),
            grid_size: int = 5,
            spline_order:int = 3,
            scale_noise:float = 0.1,
            scale_base: float = 1.0,
            scale_spline: float = 1.0,
            base_activation=torch.nn.SiLU,
            grid_eps: float = 0.02,
            grid_range: tuple = [-1, 1],
            device: str = "cpu",
            dinamic_grid = False
        ):
        """
        Kan Convolutional Layer with multiple convolutions
        
        Args:
            n_convs (int): Number of convolutions to apply
            kernel_size (tuple): Size of the kernel
            stride (tuple): Stride of the convolution
            padding (tuple): Padding of the convolution
            dilation (tuple): Dilation of the convolution
            grid_size (int): Size of the grid
            spline_order (int): Order of the spline
            scale_noise (float): Scale of the noise
            scale_base (float): Scale of the base
            scale_spline (float): Scale of the spline
            base_activation (torch.nn.Module): Activation function
            grid_eps (float): Epsilon of the grid
            grid_range (tuple): Range of the grid
            device (str): Device to use
        """


        super(KAN_Convolutional_Layer, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.device = device
        self.dilation = dilation
        self.padding = padding
        self.convs = torch.nn.ModuleList()
        self.n_convs = n_convs
        self.stride = stride
        self.dinamic_grid = dinamic_grid

        # Create n_convs KAN_Convolution objects
        for _ in range(n_convs):
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
                    device = device,
                    dinamic_grid = dinamic_grid

                )
            )

    def forward(self, x: torch.Tensor):
        # If there are multiple convolutions, apply them all
        if self.dinamic_grid and self.training:
            x_min = torch.min(x).item()
            x_max = torch.max(x).item()
            if  x_min < self.convs[0].conv.grid[0,0] or x_max > self.convs[0].conv.grid[0,-1]:
                print("updateo")
                batch_size,n_channels,n, m = x.shape
                h_out = np.floor((n + 2 * self.padding[0] - self.kernel_size[0] - (self.kernel_size[1] - 1) * (self.dilation[0] - 1)) / self.stride[0]).astype(int) + 1
                w_out = np.floor((m + 2 * self.padding[1] - self.kernel_size[0] - (self.kernel_size[1] - 1) * (self.dilation[1] - 1)) / self.stride[1]).astype(int) + 1
                unfold = torch.nn.Unfold((self.kernel_size[0],self.kernel_size[1]), dilation=self.dilation, padding=self.padding, stride=self.stride)
                conv_groups = unfold(x[:,:,:,:]).view(batch_size, n_channels,  self.kernel_size[0]*self.kernel_size[1], h_out*w_out).transpose(2, 3)#reshape((batch_size,n_channels,h_out,w_out))
                conv_groups = conv_groups.flatten(start_dim=0,end_dim = 2)
                #print("conv",conv_groups.shape)
                for conv in self.convs:
                    conv.conv.update_grid(conv_groups)
        return convolution.multiple_convs_kan_conv2d(x, self.convs,self.kernel_size[0],self.stride,self.dilation,self.padding,self.device)

class KAN_Convolution(torch.nn.Module):
    def __init__(
            self,
            kernel_size: tuple = (2,2),
            stride: tuple = (1,1),
            padding: tuple = (0,0),
            dilation: tuple = (1,1),
            grid_size: int = 5,
            spline_order: int = 3,
            scale_noise: float = 0.1,
            scale_base: float = 1.0,
            scale_spline: float = 1.0,
            base_activation=torch.nn.SiLU,
            grid_eps: float = 0.02,
            grid_range: tuple = [-1, 1],
            device = "cpu",
            dinamic_grid = False
        ):
        """
        Args
        """
        super(KAN_Convolution, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.device = device
        self.dinamic_grid = dinamic_grid

        self.conv = KANLinear(
            in_features = math.prod(kernel_size),
            out_features = 1,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
            dinamic_grid = dinamic_grid

            
        )

    def forward(self, x: torch.Tensor):
        return convolution.kan_conv2d(x, self.conv,self.kernel_size[0],self.stride,self.dilation,self.padding,self.device)
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum( layer.regularization_loss(regularize_activation, regularize_entropy) for layer in self.layers)



