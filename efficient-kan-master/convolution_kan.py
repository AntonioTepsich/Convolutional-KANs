from torch import nn
from src.efficient_kan.kan import KANLinear
class CNN_KAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flat = nn.Flatten() 
        print("shape",self.flat)
        self.kan1 = KANLinear(
                    512,
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
        #self.fc4 = nn.Linear(512, 10)
        """
        self.layers = torch.nn.ModuleList()

        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )"""
    def forward(self, x):
        # input 3x32x32, output 32x32x32
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # input 32x32x32, output 32x32x32
        x = self.act2(self.conv2(x))
        # input 32x32x32, output 32x16x16
        x = self.pool2(x)
        # input 32x16x16, output 8192
        x = self.flat(x)
        # input 8192, output 512
        x = self.kan1(x) 
        return x
    

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