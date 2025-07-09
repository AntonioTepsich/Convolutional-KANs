#!git clone -b Colab https://github.com/AntonioTepsich/Convolutional-KANs.git
# %cd Convolutional-KANs/
#!git pull
import sys

sys.path.insert(1, "Convolutional-KANs")

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
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from architectures_28x28.KKAN import *
from architectures_28x28.conv_and_kan import *
from architectures_28x28.KANConvs_MLP import *
from architectures_28x28.KANConvs_MLP_2 import *
from architectures_28x28.SimpleModels import *
from evaluations import *
from hiperparam_tuning import *

torch.manual_seed(42)  # Lets set a seed for the weights initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Transformations
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# Load MNIST and filter by classes
mnist_train = FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)

mnist_test = FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)


test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
dataset_name = "FashionMNIST_torchlike"
path = f"models/{dataset_name}"

if not os.path.exists("models"):
    os.mkdir("models")

if not os.path.exists("results"):
    os.mkdir("results")


if not os.path.exists(path):
    os.mkdir(path)

results_path = os.path.join("results", dataset_name)
if not os.path.exists(results_path):
    os.mkdir(results_path)


def train_all_kans(grid_size):

    search_hiperparams_and_get_final_model(
        KANC_MLP,
        True,
        mnist_train,
        test_loader,
        max_epochs=20,
        path=path,
        search_grid_combinations=8,
        folds=1,
        dataset_name=dataset_name,
        grid_size=grid_size,
    )
    search_hiperparams_and_get_final_model(
        KANC_MLP_Big,
        True,
        mnist_train,
        test_loader,
        max_epochs=20,
        path=path,
        search_grid_combinations=8,
        folds=1,
        dataset_name=dataset_name,
        grid_size=grid_size,
    )
    search_hiperparams_and_get_final_model(
        KANC_MLP_Medium,
        True,
        mnist_train,
        test_loader,
        max_epochs=20,
        path=path,
        search_grid_combinations=8,
        folds=1,
        dataset_name=dataset_name,
        grid_size=grid_size,
    )
    search_hiperparams_and_get_final_model(
        KKAN_Convolutional_Network,
        True,
        mnist_train,
        test_loader,
        max_epochs=20,
        path=path,
        search_grid_combinations=8,
        folds=1,
        dataset_name=dataset_name,
        grid_size=grid_size,
    )

    search_hiperparams_and_get_final_model(
        KKAN_Small,
        True,
        mnist_train,
        test_loader,
        max_epochs=20,
        path=path,
        search_grid_combinations=8,
        folds=1,
        dataset_name=dataset_name,
        grid_size=grid_size,
    )

    search_hiperparams_and_get_final_model(
        NormalConvsKAN,
        True,
        mnist_train,
        test_loader,
        max_epochs=20,
        path=path,
        search_grid_combinations=8,
        folds=1,
        dataset_name=dataset_name,
        grid_size=grid_size,
    )

    search_hiperparams_and_get_final_model(
        NormalConvsKAN_Medium,
        True,
        mnist_train,
        test_loader,
        max_epochs=20,
        path=path,
        search_grid_combinations=8,
        folds=1,
        dataset_name=dataset_name,
        grid_size=grid_size,
    )


train_all_kans(grid_size=10)
train_all_kans(grid_size=20)


search_hiperparams_and_get_final_model(
    SimpleCNN,
    False,
    mnist_train,
    test_loader,
    max_epochs=20,
    path=path,
    search_grid_combinations=8,
    folds=1,
    dataset_name=dataset_name,
)
search_hiperparams_and_get_final_model(
    MediumCNN,
    False,
    mnist_train,
    test_loader,
    max_epochs=20,
    path=path,
    search_grid_combinations=8,
    folds=1,
    dataset_name=dataset_name,
)
search_hiperparams_and_get_final_model(
    CNN_Big,
    False,
    mnist_train,
    test_loader,
    max_epochs=20,
    path=path,
    search_grid_combinations=8,
    folds=1,
    dataset_name=dataset_name,
)

search_hiperparams_and_get_final_model(
    CNN_more_convs,
    False,
    mnist_train,
    test_loader,
    max_epochs=20,
    path=path,
    search_grid_combinations=8,
    folds=1,
    dataset_name=dataset_name,
)
