from torch import nn
import sys
import torch.nn.functional as F

sys.path.append('./kan_convolutional')
#from kan_convolutional.KANConv import KAN_Convolutional_Layer

class Cifar10KANConvModel(ImageClassificationBase):
    def _init_(self,grid_size = 5):
        super()._init_()
        self.network = nn.Sequential(
            KANConv2DLayer(
            input_dim=3, output_dim=32,
            kernel_size=(3,3), spline_order=3,
            groups=1, padding=1, # padding=1?
            stride=1, dilation=1,
            grid_size=grid_size, base_activation=nn.SiLU,
            grid_range=[0,1], dropout=0.0,
        ),
            nn.ReLU(),
            KANConv2DLayer(
            input_dim=32, output_dim=64,
            kernel_size=(3,3), spline_order=3,
            groups=1, padding=1, # padding=1?
            stride=1, dilation=1,
            grid_size=grid_size, base_activation=nn.SiLU,
            grid_range=[0,1], dropout=0.0,
        ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            KANConv2DLayer(
            input_dim=64, output_dim=128,
            kernel_size=(3,3), spline_order=3,
            groups=1, padding=1, # padding=1?
            stride=1, dilation=1,
            grid_size=grid_size, base_activation=nn.SiLU,
            grid_range=[0,1], dropout=0.0,
        ),
            nn.ReLU(),
           KANConv2DLayer(
            input_dim=128, output_dim=128,
            kernel_size=(3,3), spline_order=3,
            groups=1, padding=1, # padding=1?
            stride=1, dilation=1,
            grid_size=grid_size, base_activation=nn.SiLU,
            grid_range=[0,1], dropout=0.0,
        ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            KANConv2DLayer(
            input_dim=128, output_dim=256,
            kernel_size=(3,3), spline_order=3,
            groups=1, padding=1, # padding=1?
            stride=1, dilation=1,
            grid_size=grid_size, base_activation=nn.SiLU,
            grid_range=[0,1], dropout=0.0,
        ),
            nn.ReLU(),
            KANConv2DLayer(
            input_dim=256, output_dim=256,
            kernel_size=(3,3), spline_order=3,
            groups=1, padding=1, # padding=1?
            stride=1, dilation=1,
            grid_size=grid_size, base_activation=nn.SiLU,
            grid_range=[0,1], dropout=0.0,
        ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4
            nn.Flatten(), 
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))
        
    def forward(self, xb):
        return self.network(xb)