#ESTE SCRIPT TIENE LAS CAPAS CONVOLUCIONALES NORMALES Y AL FINAL LA KAN EN VEZ DE UN MLP
from torch import nn
from KANConv import *

class KAN_Convolutional_Network(nn.Module):
    def __init__(self,device  = "cuda"):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(n_convs = 2,
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
            device = device)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flat = nn.Flatten() 
        self.kan1 = KANLinear(
                    2*169,
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
        x = self.conv1(x)
        # input 32x32x32, output 32x32x32
        # input 32x32x32, output 32x16x16
        #print(x.shape)
        x = self.pool2(x)
        # input 32x16x16, output 8192
        x = self.flat(x)
        # input 8192, output 512
        x = self.kan1(x) 
        return x