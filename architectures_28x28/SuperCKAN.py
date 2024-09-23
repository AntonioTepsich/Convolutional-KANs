from torch import nn
import sys
 
sys.path.append('./kan_convolutional')
# from kan_convolutional.KANConv import KAN_Convolutional_Layer
from kan_convolutional.kan_conv import KANConv2DLayer


class SuperCKAN(nn.Module):
    def __init__(self,device: str = 'cpu'):
        super().__init__()
        # self.conv1 = KAN_Convolutional_Layer(
        #     n_convs = 12,
        #     kernel_size= (5,5),
        #     device = device
        # )
        self.conv1 = KANConv2DLayer(
            input_dim=1,
            output_dim=12,
            kernel_size=(5,5),
            spline_order=3,
            groups=1,
            padding=0,
            stride=1,
            dilation=1,
            grid_size=10,
            base_activation=nn.SiLU,
            grid_range=[0,1],
            dropout=0.0,
        )

        # self.conv2 = KAN_Convolutional_Layer(
        #     n_convs = 12,
        #     kernel_size = (4,4),
        #     device = device,
        #     grid_size=10
        # )
        self.conv2 = KANConv2DLayer(
            input_dim=12,
            output_dim=144,
            kernel_size=(4,4),
            spline_order=3,
            groups=1,
            padding=0,
            stride=1,
            dilation=1,
            grid_size=10,
            base_activation=nn.SiLU,
            grid_range=[0,1],
            dropout=0.0,
        )
        # self.conv3 = KAN_Convolutional_Layer(
        #     n_convs = 24,
        #     kernel_size = (3,3),
        #     device = device,
        #     grid_size=10
        # )

        self.conv3 = KANConv2DLayer(
            input_dim=144,
            output_dim=3456,
            kernel_size=(3,3),
            spline_order=3,
            groups=1,
            padding=0,
            stride=1,
            dilation=1,
            grid_size=10,
            base_activation=nn.SiLU,
            grid_range=[0,1],
            dropout=0.0,
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )
        
        self.flat = nn.Flatten() 
        self.linear1 = nn.Linear(2304, 256)
        self.dropout1 = nn.Dropout(0.25)

        self.linear2 = nn.Linear(256, 10)


    def forward(self, x):
        x = self.conv1(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.flat(x)
        x = self.linear1(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        return x
