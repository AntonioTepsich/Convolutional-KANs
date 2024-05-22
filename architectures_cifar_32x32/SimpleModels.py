from torch import nn
import sys
import torch.nn.functional as F

# directory reach
sys.path.append('./kan_convolutional')

class SimpleCNN_Small(nn.Module):
    def __init__(self):
        super(SimpleCNN_Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 15, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(16, 16*4, kernel_size=4, padding=1)
        self.conv3 = nn.Conv2d(16*4, 16*4*3, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4096, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x
class SimpleCNN_Medium(nn.Module):
    def __init__(self):
        super(SimpleCNN_Medium, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(5, 25, kernel_size=3, padding=0)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(625, 245)
        self.fc2 = nn.Linear(245, 10)
        self.name = "CNN (Medium)"
class SimpleLinear(nn.Module):
    def __init__(self):
        super(SimpleLinear, self).__init__()
        self.linear = nn.Linear(3072, 10)
        self.flatten = nn.Flatten()


    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)

        return x
