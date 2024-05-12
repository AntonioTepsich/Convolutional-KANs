from torch import nn
import sys
import torch.nn.functional as F

sys.path.append('../kan_convolutional')

from KANLinear import KANLinear

class NormalConvsKAN(nn.Module):
    def __init__(self):
        super(NormalConvsKAN, self).__init__()
        # Capa convolucional, asumiendo una entrada con 1 canal (imagen en escala de grises)
        # y produciendo 16 canales de salida, con un kernel de tamaño 3x3
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3, padding=1)

        # Capa de MaxPooling, utilizando un kernel de tamaño 2x2
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        # Capa lineal (Fully Connected)
        # Suponiendo una entrada de imagen de tamaño 28x28, después de conv y maxpool, será 14x14
        self.flatten = nn.Flatten()
        self.kan1 = KANLinear(
            245,
            10,
            grid_size=10,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0,1])


    def forward(self, x):
        # Aplicar la capa convolucional seguida de activación ReLU
        x = F.relu(self.conv1(x))
        # Aplicar max pooling
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        # Aplicar max pooling
        x = self.maxpool(x)
        # Aplanar los datos
        x = self.flatten(x)
        # Capa lineal con 10 salidas
        x = self.kan1(x)
        x = F.log_softmax(x, dim=1)

        return x