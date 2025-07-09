import sys

sys.path.insert(1, "Convolutional-KANs")

import os
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from architectures_28x28.KKAN import *
from architectures_28x28.conv_and_kan import *

from architectures_28x28.KANConvs_MLP import *
from architectures_28x28.KANConvs_MLP_2 import *
from architectures_28x28.SimpleModels import *
from evaluations import *
import time


# from hiperparam_tuning import *
# from calflops import calculate_flops
def calculate_time(model, train_obj, test_obj, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    criterion = nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(
        train_obj, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_obj, batch_size=batch_size, shuffle=True
    )
    start = time.perf_counter()

    train_and_test_models(
        model,
        device,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        epochs=1,
        scheduler=scheduler,
        path=None,
        verbose=False,
        save_last=False,
        patience=np.inf,
    )
    total_time = time.perf_counter() - start
    print(model.name, "took:", total_time)
    return total_time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Transformations
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# Load MNIST and filter by classes
mnist_train = MNIST(root="./data", train=True, download=True, transform=transform)

mnist_test = MNIST(root="./data", train=False, download=True, transform=transform)


dataset_name = "MNIST"
path = f"models/{dataset_name}"

if not os.path.exists("results"):
    os.mkdir("results")

if not os.path.exists(path):
    os.mkdir(path)

results_path = os.path.join("results", dataset_name)
if not os.path.exists(results_path):
    os.mkdir(results_path)

batch_size = 128
models = [
    KANC_MLP(grid_size=10),
    KANC_MLP_Medium(grid_size=10),
    KANC_MLP_Big(grid_size=10),
    KANC_MLP(grid_size=20),
    KANC_MLP_Medium(grid_size=20),
    KANC_MLP_Big(grid_size=20),
    SimpleCNN(),
    MediumCNN(),
    CNN_Big(),
    CNN_more_convs(),
    KKAN_Convolutional_Network(grid_size=10),
    KKAN_Small(grid_size=10),
    NormalConvsKAN(grid_size=10),
    NormalConvsKAN_Medium(grid_size=10),
    KKAN_Convolutional_Network(grid_size=20),
    KKAN_Small(grid_size=20),
    NormalConvsKAN(grid_size=20),
    NormalConvsKAN_Medium(grid_size=20),
]

import json

dictionary = {}
for m in models:
    t = calculate_time(m, mnist_train, mnist_test, batch_size)
    dictionary[m.name] = t
with open(f"results/{dataset_name}/epoch_times.json", "w") as outfile:
    json.dump(dictionary, outfile)
