from torch import nn
import sys
import torch.nn.functional as F

# sys.path.append('../kan_convolutional')
from kan_convolutional.KANConv import KAN_Convolutional_Layer

class KANC_MLP(nn.Module):
    def __init__(self,device: str = 'cpu'):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(
            n_convs = 5,
            kernel_size= (3,3),
            device = device
        )

        self.conv2 = KAN_Convolutional_Layer(
            n_convs = 5,
            kernel_size = (3,3),
            device = device,
            dinamic_grid=True,
            grid_size= 10

        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )
        
        self.flat = nn.Flatten() 
        
        #self.linear1 = nn.Linear(625, 256)
        self.linear2 = nn.Linear(625, 10)
        self.name = "KAN Conv Grid updated & 1 Layer MLP"


    def forward(self, x):
        x = self.conv1(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool1(x)
        x = self.flat(x)
        x = self.linear2(x)
        x = F.log_softmax(x, dim=1)
        return x

class KANC_MLP_sin_grid(nn.Module):
    def __init__(self,device: str = 'cpu'):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(
            n_convs = 5,
            kernel_size= (3,3),
            device = device
        )

        self.conv2 = KAN_Convolutional_Layer(
            n_convs = 5,
            kernel_size = (3,3),
            device = device,
            dinamic_grid=False,
            grid_size= 10   

        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )
        
        self.flat = nn.Flatten() 
        
        #self.linear1 = nn.Linear(625, 256)
        self.linear2 = nn.Linear(625, 10)
        self.name = "KAN Conv & 1 Layer MLP"

    def forward(self, x):
        x = self.conv1(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool1(x)
        x = self.flat(x)
        x = self.linear2(x)
        x = F.log_softmax(x, dim=1)
        return x