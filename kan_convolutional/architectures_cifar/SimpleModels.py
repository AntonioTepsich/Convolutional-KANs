from torch import nn
import sys
import torch.nn.functional as F

# directory reach
sys.path.append('../kan_convolutional')

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Capa convolucional, asumiendo una entrada con 1 canal (imagen en escala de grises)
        # y produciendo 16 canales de salida, con un kernel de tamaño 3x3
        self.conv1 = nn.Conv2d(3, 16, kernel_size=2, padding=1)
        
        # Capa de MaxPooling, utilizando un kernel de tamaño 2x2
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        # Capa lineal (Fully Connected)
        # Suponiendo una entrada de imagen de tamaño 28x28, después de conv y maxpool, será 14x14
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4096, 10)

    def forward(self, x):
        # Aplicar la capa convolucional seguida de activación ReLU
        x = F.relu(self.conv1(x))
        # Aplicar max pooling
        x = self.maxpool(x)
        # Aplanar los datos
        x = self.flatten(x)
        # Capa lineal con 10 salidas
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x

class SimpleLinear(nn.Module):
    def __init__(self):
        super(SimpleLinear, self).__init__()
        # Definir una sola capa lineal con 'input_features' entradas y 10 salidas
        self.linear = nn.Linear(3072, 10)
        self.flatten = nn.Flatten()


    def forward(self, x):
        # Pasar los datos a través de la capa lineal
        x = self.flatten(x)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)

        return x
