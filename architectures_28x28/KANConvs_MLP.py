from torch import nn
import sys
import torch.nn.functional as F

sys.path.append('./kan_convolutional')
#from kan_convolutional.KANConv import KAN_Convolutional_Layer
from kan_convolutional.kan_conv import KANConv2DLayer

from new.convkan.convkan_layer import ConvKAN

class KANC_MLP(nn.Module):
    def __init__(self,grid_size: int = 5):
        self.name = "KAN Conv & 1 Layer MLP"

        super().__init__()
        # self.conv1 = KAN_Convolutional_Layer(
        #     n_convs = 5,
        #     kernel_size= (3,3),
        #     device = device
        # )

        self.conv1 = ConvKAN(
            in_channels=1,
            out_channels=5,
            kernel_size=(3,3),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            padding_mode="zeros",
            bias=True,
            grid_size=grid_size,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=(-1, 1),
            kan_type="EfficientKAN",
        )


        # self.conv2 = KAN_Convolutional_Layer(
        #     n_convs = 5,
        #     kernel_size = (3,3),
        #     device = device
        # )
        self.conv2 = ConvKAN(
            in_channels=5,
            out_channels=25,
            kernel_size=(3,3),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            padding_mode="zeros",
            bias=True,
            grid_size=grid_size,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=(-1, 1),
            kan_type="EfficientKAN",
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )
        
        self.flat = nn.Flatten() 
        
        self.linear1 = nn.Linear(625, 256)
        self.linear2 = nn.Linear(256, 10)


    def forward(self, x):
        x = self.conv1(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool1(x)
        x = self.flat(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = F.log_softmax(x, dim=1)
        return x
