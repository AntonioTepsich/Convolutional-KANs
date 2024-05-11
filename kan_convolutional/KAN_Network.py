#ESTE SCRIPT TIENE LAS CAPAS CONVOLUCIONALES NORMALES Y AL FINAL LA KAN EN VEZ DE UN MLP
from torch import nn
from KANConv import *


class KAN_Convolutional_Network(nn.Module):
    def __init__(self,device  = "cuda"):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(
            n_convs = 2,
            kernel_size= (2,2),
            stride =  (1,1),
            padding=(0,0),
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
            device = device
        )

        self.conv2 = KAN_Convolutional_Layer(
            n_convs = 2,
            kernel_size= (2,2),
            stride =  (1,1),
            padding=(0,0),
            grid_size=7,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
            device = device
        )

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.flat = nn.Flatten() 
        
        self.kan1 = KANLinear(
            144,
            20,
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
        x = self.conv1(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool1(x)
        x = self.flat(x)
    
        x = self.kan1(x) 
        return x
    