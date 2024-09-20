#!git clone -b Colab https://github.com/AntonioTepsich/Convolutional-KANs.git
#%cd Convolutional-KANs/
#!git pull
import sys
sys.path.insert(1,'Convolutional-KANs')

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from architectures_28x28.KKAN import KKAN_Convolutional_Network
from architectures_28x28.conv_and_kan import NormalConvsKAN
from architectures_28x28.CKAN_BN import CKAN_BN
from architectures_28x28.KANConvs_MLP import KANC_MLP,KANC_MLP_sin_grid
from architectures_28x28.KANConvs_MLP_2 import KANC_MLP_2,KANC_MLP_sin_grid_2
from architectures_28x28.SimpleModels import *
from architectures_28x28.ConvNet import ConvNet
from evaluations import *
from hiperparam_tuning import *
torch.manual_seed(42) #Lets set a seed for the weights initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Transformaciones
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Cargar MNIST y filtrar por dos clases
mnist_train = MNIST(root='./data', train=True, download=True, transform=transform)

mnist_test = MNIST(root='./data', train=False, download=True, transform=transform)

# DataLoader
train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
dataset_name = "MNIST"
path = f"models/{dataset_name}"
if not os.path.exists(path):
    os.mkdir("/".join(path.split("/")[:-1]))
    os.mkdir(path)
model_KANC_MLP= KANC_MLP()
results_path = os.path.join("results",dataset_name)
if not os.path.exists(results_path):
    os.mkdir(results_path)
search_hiperparams_and_get_final_model(KANC_MLP,True, mnist_train,  test_loader,max_epochs= 20,path = path,search_grid_combinations = 10 ,folds = 3,dataset_name=dataset_name)

#model_KANC_MLP_sin_grid= KANC_MLP_sin_grid(device = device)

#search_hiperparams_and_get_final_model(model_KANC_MLP_sin_grid,True, mnist_train,  test_loader,max_epochs= 20,path = path,search_grid_combinations = 10 ,folds = 3,dataset_name=dataset_name)

#Bigger
model_KANC_MLP_2= KANC_MLP_2(device = device)
search_hiperparams_and_get_final_model(model_KANC_MLP_2,True, mnist_train,  test_loader,max_epochs= 20,path = path,search_grid_combinations = 10 ,folds = 3,dataset_name=dataset_name)

#model_KANC_MLP_sin_grid_2= KANC_MLP_sin_grid_2(device = device)
#search_hiperparams_and_get_final_model(model_KANC_MLP_sin_grid_2,True, mnist_train,  test_loader,max_epochs= 20,path = path,search_grid_combinations = 10 ,folds = 3,dataset_name=dataset_name)

model_CKAN_BN= CKAN_BN(device)
search_hiperparams_and_get_final_model(model_CKAN_BN,True, mnist_train,  test_loader,max_epochs= 20,path = path,search_grid_combinations = 10 ,folds = 3,dataset_name=dataset_name)

model_KKAN_Convolutional_Network = KKAN_Convolutional_Network(device = device)
search_hiperparams_and_get_final_model(model_KKAN_Convolutional_Network,True, mnist_train,  test_loader,max_epochs= 20,path = path,search_grid_combinations = 10 ,folds = 3,dataset_name=dataset_name)

model_Convs_and_KAN= NormalConvsKAN()
search_hiperparams_and_get_final_model(model_Convs_and_KAN,True, mnist_train,  test_loader,max_epochs= 20,path = path,search_grid_combinations = 10 ,folds = 3,dataset_name=dataset_name)

model_ConvNet = ConvNet()
search_hiperparams_and_get_final_model(model_ConvNet,False, mnist_train,  test_loader,max_epochs= 20,path = path,search_grid_combinations = 10 ,folds = 3,dataset_name=dataset_name)

model_SimpleCNN = SimpleCNN()
search_hiperparams_and_get_final_model(model_SimpleCNN,False, mnist_train,  test_loader,max_epochs= 20,path = path,search_grid_combinations = 10 ,folds = 3,dataset_name=dataset_name)

model_SimpleCNN_2 = SimpleCNN_2()
search_hiperparams_and_get_final_model(model_SimpleCNN_2,False, mnist_train,  test_loader,max_epochs= 20,path = path,search_grid_combinations = 10 ,folds = 3,dataset_name=dataset_name)

model_SimpleLinear = SimpleLinear()
search_hiperparams_and_get_final_model(model_SimpleLinear,False, mnist_train,  test_loader,max_epochs= 20,path = path,search_grid_combinations = 10 ,folds = 3,dataset_name=dataset_name)
