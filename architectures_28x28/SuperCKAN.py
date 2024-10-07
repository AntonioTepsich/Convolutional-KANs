from torch import nn
import sys
 
sys.path.append('./kan_convolutional')
# from kan_convolutional.KANConv import KAN_Convolutional_Layer
from kan_convolutional.KANConv import KAN_Convolutional_Layer


class SuperCKAN(nn.Module):
    def __init__(self, grid_size: int = 5):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(
            n_convs = 12,
            kernel_size= (5,5),
            grid_size = grid_size
        )

        self.conv2 = KAN_Convolutional_Layer(
            n_convs = 12,
            kernel_size = (4,4),
            grid_size = grid_size
        )
        self.conv3 = KAN_Convolutional_Layer(
            n_convs = 24,
            kernel_size = (3,3),
            grid_size = grid_size
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