from torch import nn
from src.efficient_kan.kan import KANLinear
#ESTE SCRIPT TIENE LAS CAPAS CONVOLUCIONALES NORMALES Y AL FINAL LA KAN EN VEZ DE UN MLP


class CNN_KAN(nn.Module):
    def __init__(self):
        super(CNN_KAN, self).__init__()
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
            KANLinear(
                in_features=32*7*7, 
                out_features=10,
                grid_size=10,
                spline_order=3,
                scale_noise=0.01,
                scale_base=1,
                scale_spline=1,
                base_activation=nn.SiLU,
                grid_eps=0.02,
                grid_range=[0,1],

            ),
        )

    def forward(self, x):
        return self.model(x)