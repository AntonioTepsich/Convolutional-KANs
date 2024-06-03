from torch import nn
import sys
import torch.nn.functional as F

sys.path.append('./kan_convolutional')

from kan_convolutional.KANConv import KAN_Convolutional_Layer
from kan_convolutional.KANLinear import KANLinear
class KKAN_Small(nn.Module):
    def __init__(self,device: str = 'cpu'):
        super().__init__()
        self.name="KKAN (Small)"
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
        self.kan1 = KANLinear(
            192 ,
            10,
            grid_size=15,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0,1],
            dinamic_grid= True
        )
    def regularization_loss(self,):
        self.conv1.regularization_loss() + self.conv2.regularization_loss() +self.kan1.regularization_loss() 
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.flat(x)
        x = self.kan1(x)
        x = F.log_softmax(x, dim=1)
        return x
class KKAN_Medium(nn.Module):
    def __init__(self,device: str = 'cpu'):
        super().__init__()
        self.name="KKAN (Medium)"

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
        
        
        self.kan1 = KANLinear(
            384    ,
            150,
            grid_size=15,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0,1],
            dinamic_grid= True
        )
        self.kan2 = KANLinear(
            150    ,
            10,
            grid_size=10,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0,1],
            dinamic_grid= True
        )


    def forward(self, x):
        x = self.conv1(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.flat(x)

        x = self.kan1(x)
        x = self.kan2(x)
        x = F.log_softmax(x, dim=1)

        return x
    

class KKAN_Convolutional_Network_big(nn.Module):
    def __init__(self,device: str = 'cpu'):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(
            n_convs = 10,
            kernel_size= (4,4),
            device = device
        )

        self.conv2 = KAN_Convolutional_Layer(
            n_convs = 10,
            kernel_size = (3,3),
            device = device
        )
        self.conv3 = KAN_Convolutional_Layer(
            n_convs = 5,
            kernel_size = (3,3),
            device = device
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )
        
        self.flat = nn.Flatten() 
        

        self.kan1 = KANLinear(
            1500    ,
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
        x = self.conv1(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.flat(x)

        x = self.kan1(x) 
        x = F.log_softmax(x, dim=1)

        return x 