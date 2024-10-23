#!git clone -b Colab https://github.com/AntonioTepsich/Convolutional-KANs.git
#%cd Convolutional-KANs/
#!git pull
import sys
sys.path.insert(1,'Convolutional-KANs')
sys.path.append('../CKAN')
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from architectures_28x28.KKAN import KKAN_Convolutional_Network,KKAN_Ultra_Small
from architectures_28x28.conv_and_kan import NormalConvsKAN
from architectures_28x28.CKAN_BN import CKAN_BN
from architectures_28x28.KANConvs_MLP import KANC_MLP
from architectures_28x28.KANConvs_MLP_2 import KANC_MLP_2
from architectures_28x28.SimpleModels import *
from architectures_28x28.ConvNet import ConvNet
from evaluations import *
from hiperparam_tuning import *
torch.manual_seed(42) #Lets set a seed for the weights initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

#Load MNIST and filter by classes
mnist_train = MNIST(root='./data', train=True, download=True, transform=transform)

mnist_test = MNIST(root='./data', train=False, download=True, transform=transform)
config= {'lr': 0.0005, 'weight_decay': 1e-05, 'batch_size': 128, 'grid_size': 20}
DataLoader
train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
dataset_name = "MNIST"
path = f"models/{dataset_name}"

if not os.path.exists("models"):
    os.mkdir("models")

if not os.path.exists("results"):
    os.mkdir("results")

if not os.path.exists(path):
    os.mkdir(path)

model_KANC_MLP= KANC_MLP()
results_path = os.path.join("results",dataset_name)
if not os.path.exists(results_path):
    os.mkdir(results_path)

train_tune(config,KANC_MLP,True, mnist_train,  epochs = 1,folds= 1,profile =True)


train_tune(config,KANC_MLP_2,True, mnist_train,  epochs = 1,folds= 1,profile =True)

train_tune(config,CKAN_BN,True, mnist_train,  epochs = 1,folds= 1,profile =True)

train_tune(config,KKAN_Convolutional_Network,True, mnist_train,  epochs = 1,folds= 1,profile =True)

train_tune(config,NormalConvsKAN,True, mnist_train,  epochs = 1,folds= 1,profile =True)

train_tune(config,ConvNet,True, mnist_train,  epochs = 1,folds= 1,profile =True)
train_tune(config,SimpleCNN,True, mnist_train,  epochs = 1,folds= 1,profile =True)
train_tune(config,SimpleCNN_2,True, mnist_train,  epochs = 1,folds= 1,profile =True)
train_tune(config,SimpleLinear,True, mnist_train,  epochs = 1,folds= 1,profile =True)

train_tune(config,newMediumCNN,True, mnist_train,  epochs = 1,folds= 1,profile =True)
train_tune(config,KKAN_Ultra_Small,True, mnist_train,  epochs = 1,folds= 1,profile =True)
