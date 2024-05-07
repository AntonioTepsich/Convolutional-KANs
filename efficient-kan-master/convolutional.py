
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Definir la arquitectura utilizando nn.Sequential
        self.model = nn.Sequential(
            # Primera capa convolucional
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Segunda capa convolucional
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Capa completamente conectada (FC)
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),  # Ajusta esta dimensión a la salida de las capas de pooling
            nn.ReLU(),
            nn.Linear(128, 10)  # Número de clases en la salida (por ejemplo, 10 para MNIST)
        )

    def forward(self, x):
        return self.model(x)