from torch import nn
import sys
import torch.nn.functional as F

sys.path.append('./kan_convolutional')
from KANConv import KAN_Convolutional_Layer

class CKAN(nn.Module):
    def __init__(self,device: str = 'cpu'):
        super().__init__()
        self.name="KANConv 2 layers MLP (Small)"
        self.conv1 = KAN_Convolutional_Layer(
            n_convs = 4,
            kernel_size= (5,5),
            device = device
        )

        self.conv2 = KAN_Convolutional_Layer(
            n_convs = 4,
            kernel_size = (4,4),
            device = device,
            grid_size = 10,
            dinamic_grid = True
        )

        self.conv3 = KAN_Convolutional_Layer(
            n_convs = 4,
            kernel_size = (3,3),
            device = device,
            grid_size = 10,
            dinamic_grid = True
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )
        
        self.flat = nn.Flatten() 
        
        self.linear1 = nn.Linear(192, 256)
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
        x = self.linear2(x)
        x = F.log_softmax(x, dim=1)
        return x
    
class CKAN_Medium(nn.Module):
    def __init__(self,device: str = 'cpu'):
        super().__init__()
        self.name="KANConv 2 layers MLP (Medium)"
        self.conv1 = KAN_Convolutional_Layer(
            n_convs = 8,
            kernel_size= (5,5),
            device = device
        )

        self.conv2 = KAN_Convolutional_Layer(
            n_convs = 4,
            kernel_size = (4,4),
            device = device,
            grid_size = 10,
            dinamic_grid = True
        )

        self.conv3 = KAN_Convolutional_Layer(
            n_convs = 4,
            kernel_size = (3,3),
            device = device,
            grid_size = 10,
            dinamic_grid = True
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )
        
        self.flat = nn.Flatten() 
        
        self.linear1 = nn.Linear(384, 256)
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
        x = self.linear2(x)
        x = F.log_softmax(x, dim=1)
        return x
    

class CKAN_Big(nn.Module):
    def __init__(self,device: str = 'cpu'):
        super().__init__()
        self.name="KANConv 2 layers MLP (Big)"
        self.conv1 = KAN_Convolutional_Layer(
            n_convs = 16,
            kernel_size= (5,5),
            device = device
        )

        self.conv2 = KAN_Convolutional_Layer(
            n_convs = 8,
            kernel_size = (4,4),
            device = device,
            grid_size = 10,
            dinamic_grid = True
        )

        self.conv3 = KAN_Convolutional_Layer(
            n_convs = 4,
            kernel_size = (4,4),
            device = device,
            grid_size = 15,
            dinamic_grid = True
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )
        
        self.flat = nn.Flatten() 
        
        self.linear1 = nn.Linear(1536, 256)
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
        x = self.linear2(x)
        x = F.log_softmax(x, dim=1)
        return x