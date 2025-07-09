from torch import nn
import sys
import torch.nn.functional as F

sys.path.append("./kan_convolutional")

from kan_convolutional.KANLinear import KANLinear


class NormalConvsKAN(nn.Module):
    def __init__(self, grid_size=5):
        super(NormalConvsKAN, self).__init__()
        # Convolutional layer, assuming an input with 1 channel (grayscale image)
        # and producing 16 output channels, with a kernel size of 3x3
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, padding=(0, 0))
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3, padding=(0, 0))
        self.name = f"Conv & KAN (Small) (gs = {grid_size})"

        # Max pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        # Flatten layer
        self.flatten = nn.Flatten()

        # KAN layer
        self.kan1 = KANLinear(
            125,
            10,
            grid_size=grid_size,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0, 1],
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.kan1(x)
        x = F.log_softmax(x, dim=1)

        return x


class NormalConvsKAN_Medium(nn.Module):
    def __init__(self, grid_size=5):
        super(NormalConvsKAN_Medium, self).__init__()

        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, padding=(0, 0))
        self.conv2 = nn.Conv2d(5, 10, kernel_size=3, padding=(0, 0))
        self.name = f"Conv & KAN (Medium) (gs = {grid_size})"

        # Max pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        # Flatten layer
        self.flatten = nn.Flatten()

        # KAN layer
        self.kan1 = KANLinear(
            250,
            10,
            grid_size=grid_size,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0, 1],
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.kan1(x)
        x = F.log_softmax(x, dim=1)

        return x
