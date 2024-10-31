from architectures_28x28.KANConvs_MLP import *
from architectures_28x28.KANConvs_MLP_2 import *
from architectures_28x28.SimpleModels import *
from evaluations import *
from hiperparam_tuning import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torchvision.datasets import FashionMNIST
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Transformaciones
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

mnist_train = FashionMNIST(root='./data', train=True, download=True, transform=transform)

mnist_test = FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

get_best_model(KANC_MLP_Big,epochs=20,config = {'lr': 0.0005, 'weight_decay': 1e-05, 'batch_size': 128}, train_obj= mnist_train,test_loader= test_loader,path ="FashionMNIST" ,is_kan= True,grid_size = 20)
