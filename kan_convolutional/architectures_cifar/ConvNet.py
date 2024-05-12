from torch import nn
import torch
import torch.nn.functional as F
import sys
# directory reach
sys.path.append('../kan_convolutional')

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Primera capa convolucional y segunda con el mismo padding y tamaño de kernel
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding='same')
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding='same')

        # Segunda serie de capas convolucionales
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding='same')

        # Capas de MaxPooling
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Capas de Dropout
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.5)

        # Capa completamente conectada
        self.fc1 = nn.Linear(4096, 256)  # 64 canales, tamaño de imagen reducido a 7x7
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Aplicar las primeras capas convolucionales y pooling
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.dropout1(x)

        # Aplicar las segundas capas convolucionales y pooling
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = self.dropout2(x)

        # Aplanar los datos para la entrada a las capas densas
        x = torch.flatten(x, 1)  # Aplanar todo excepto el batch
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)

        # Aplicar softmax para obtener las probabilidades de las clases
        x = F.log_softmax(x, dim=1)
        return x