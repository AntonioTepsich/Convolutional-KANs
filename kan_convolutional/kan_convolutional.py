import numpy as np
import sys
import torch
import torch.nn.functional as F
import math

import convolution
class KAN_Convolution(torch.nn.Module):
    def __init__(
        self,
        in_features = (28,28),
        kernel_size= (2,2),
        n_convs= 1,
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
        super(KAN_Convolution, self).__init__()
        self.in_features = in_features
        self.kernel_size = kernel_size
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .unsqueeze(0).unsqueeze(0).expand(*kernel_size, -1) 
            .contiguous()
        )
        print("Grid size",grid.shape)
        self.register_buffer("grid", grid)
        self.base_weight = torch.nn.Parameter(torch.Tensor(*kernel_size)) #es el w. Tenemos en cuenta 1 sola convolucion
        self.spline_weight = torch.nn.Parameter( #es el ci
            torch.Tensor(*kernel_size, grid_size + spline_order) #esta flateneado, osea que en vez de ser 2x2xc_is es 4xcis
        )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        print("dim base",self.base_weight.shape)
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        print("hola",self.b_splines(torch.Tensor([[1,3],[2,3]])))
        with torch.no_grad():
            noise = (
                (
                    torch.rand(*self.kernel_size,self.grid_size + 1)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(self.curve2coeff(
                    self.grid[:,:,:],
                    noise,
                )
            )

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        #assert x.dim() == 2 and x.size(1) == self.in_features #hacer esto bien

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        #x = x.unsqueeze(-1)
        bases = ((x >= grid[:,:, :-1]) & (x < grid[:,:, 1:])) .to(x.dtype) #ver a que constante tiene queq ser n

        for k in range(1, self.spline_order + 1):
            print("bases",bases.shape)
            bases = (
                (x - grid[:, :, : -(k + 1)])
                / (grid[:,:, k:-1] - grid[:,:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, :,k + 1 :] - x)
                / (grid[:,:, k + 1 :] - grid[:,:, 1:(-k)])
                * bases[:, :, 1:]
            )
        # assert bases.size() == (
        #     x.size(0),
        #     self.in_features,
        #     self.grid_size + self.spline_order,
        # )
        return bases.contiguous()
    def compare_grid_submatrix(self,submatrix,start,end,comparison='ge'):
        """Comparison: ge = greater or equal
        l = less than"""
        ret = []
        if comparison == 'l':
            for ti in range(start,end):
                ret.append(submatrix[ti]<self.grid[:,:,ti])
        elif comparison == 'ge':
            for ti in range(len(self.grid[0,0])):
                ret.append(submatrix[ti]>=self.grid[:,:,ti])
        return ret
    def b_splines_custom(self, x: torch.Tensor):
            """
            Compute the B-spline bases for the given input tensor.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, in_features).

            Returns:
                torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
            """
            #assert x.dim() == 2 and x.size(1) == self.in_features #hacer esto bien
            import torch.nn.functional as F

            grid: torch.Tensor = (
                self.grid
            )  # (in_features, grid_size + 2 * spline_order + 1)
            #x = x.unsqueeze(-1)
            #bases_ = torch.zeros_like(self.spline_weight)
            print("shape x",x.shape)
            bases = ((x >= grid[:,:, :-1]) & (x < grid[:,:, 1:])) .to(x.dtype) #ver a que constante tiene queq ser n
            #bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype) #el padding deberia ser el minimo y el maximo

            for k in range(1, self.spline_order + 1):
                print("bases",bases.shape)
                bases = (
                    (x - grid[:, :, : -(k + 1)])
                    / (grid[:,:, k:-1] - grid[:,:, : -(k + 1)])
                    * bases[:, :, :-1]
                ) + (
                    (grid[:, :,k + 1 :] - x)
                    / (grid[:,:, k + 1 :] - grid[:,:, 1:(-k)])
                    * bases[:, :, 1:]
                )
    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        # assert x.dim() == 2 and x.size(1) == self.in_features
        # assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        # assert result.size() == (
        #     self.out_features,
        #     self.in_features,
        #     self.grid_size + self.spline_order,
        # )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor,update_grid = False):
        #Ver bien tema dimensiones
        #assert x.dim() == 2 and x.size(1) == self.in_features

        #base_output = F.linear(self.base_activation(x), self.base_weight) #el b seria una matrix con b aplicado a cada uno de los pixeles
        # spline_output = F.linear(
        #     self.b_splines(x).view(x.size(0), -1),
        #     self.scaled_spline_weight.view(self.out_features, -1),
        # )
        # return base_output + spline_output
        if update_grid:
            self.update_grid(x)
        print("spl weig",self.spline_weight.view(-1,*self.kernel_size,-1).shape)
        splines_cis_as_matrix =  self.spline_weight.view(-1,*self.kernel_size,-1)

        kernel  = torch.dstack((splines_cis_as_matrix, self.base_weight.view(*self.kernel_size))) 
        return convolution.apply_filter_to_image(x, kernel,self.b_splines,torch.nn.SiLU, rgb = False)
    # def splines_to_matrix(self):
    #     return [i.view(self.kernel_size[0],self.kernel_size[1]) for i in self.b_splines]
    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

from src.efficient_kan.kan import KANLinear
from torch import nn

#ESTE SCRIPT TIENE LAS CAPAS CONVOLUCIONALES NORMALES Y AL FINAL LA KAN EN VEZ DE UN MLP
class KAN_Convolutional_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = KAN_Convolution(in_features = (28,28),
            kernel_size= (2,2),
            n_convs= 1,
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
            grid_range=[-1, 1],)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flat = nn.Flatten() 
        print("shape",self.flat)
        self.kan1 = KANLinear(
                    512,
                    10,
                    grid_size=10,
                    spline_order=3,
                    scale_noise=0.01,
                    scale_base=1,
                    scale_spline=1,
                    base_activation=nn.SiLU,
                    grid_eps=0.02,
                    grid_range=[0,1],
                )

    def forward(self, x):
        # input 3x32x32, output 32x32x32
        x = self.conv1(self.conv1(x))
        # input 32x32x32, output 32x32x32
        # input 32x32x32, output 32x16x16
        x = self.pool2(x)
        # input 32x16x16, output 8192
        x = self.flat(x)
        # input 8192, output 512
        x = self.kan1(x) 
        return x
    

class CNN_KAN(nn.Module):
    def __init__(self):
        super(CNN_KAN, self).__init__()
        # Definir la arquitectura utilizando nn.Sequential
        self.model = nn.Sequential(
            # Primera capa convolucional
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Segunda capa convolucional
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Capa completamente conectada (FC)
            nn.Flatten(),
            KANLinear(
                in_features=32*7*7, 
                out_features=10,
                grid_size=10,
                spline_order=3,
                scale_noise=0.01,
                scale_base=1,
                scale_spline=1,
                base_activation=nn.SiLU,
                grid_eps=0.02,
                grid_range=[0,1],

            ),
        )

    def forward(self, x):
        return self.model(x)
"""class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
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

"""
