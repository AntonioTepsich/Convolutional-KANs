
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
from evaluations import train_and_test_models
from sklearn.model_selection import KFold

def tune_hipers(model_class, is_kan, train_obj, max_epochs, n_combs , grid,folds = 3 ):
    combinations = select_hipers_randomly(grid, n_combs,seed = 42)
    best_trial = {"accuracy": 0}
    for comb in combinations:
        loss,accuracy,epochs = train_tune(comb,model_class, is_kan,train_obj=train_obj,epochs = max_epochs,folds =folds)
        if best_trial["accuracy"]<accuracy:
            best_trial["accuracy"] = accuracy
            best_trial["epochs"] = epochs
            best_trial["loss"] = loss
        print(f"Finished Trial with Hipers {comb} and got accuracy {accuracy} with epochs {epochs}")
    # Get the best trial
    print(f"Best trial config: {best_trial}")
    print(f"Best trial final validation loss: {best_trial['loss']}")
    print(f"Best trial final validation accuracy: {best_trial['accuracy']}")
    print(f"Best trial final number of epochs: {best_trial['epochs']}")
    return best_trial#best_trial.last_result['epochs'], best_trial.last_result['accuracy']


def train_tune(config,model_class, is_kan,train_obj=None,epochs = 20,folds= 3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42) #Lets set a seed for the weights initialization

    kfold = KFold(n_splits=folds, shuffle=True)
    accuracys = []
    losses = []
    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_obj)):
        if is_kan:
            model = model_class(grid_size = config["grid_size"])
        else:
            model = model_class()
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        # Define data loaders for training and testing data in this fold
        train_loader = torch.utils.data.DataLoader(
                        train_obj, 
                        batch_size=config["batch_size"], sampler=train_subsampler)
        valid_loader = torch.utils.data.DataLoader(
                        train_obj,
                        batch_size=config["batch_size"], sampler=valid_subsampler)
        
            # Init the neural network
        print("config",config)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"],weight_decay = config["weight_decay"])

        all_train_loss, all_test_loss, all_test_accuracy, all_test_precision, all_test_recall, all_test_f1 = train_and_test_models(model, device, train_loader, valid_loader, optimizer, criterion, epochs=epochs, scheduler=None,path= None,verbose= False)
        #best_epochs = np.argmin(all_test_accuracy)
        #best_accuracy = all_test_accuracy[best_epochs]
        #best_loss = all_test_loss[best_epochs]

        accuracys.append(all_test_accuracy)
        losses.append(all_test_loss)
    accuracy_per_epoch = np.mean(accuracys,axis = 0)
    loss_per_epoch = np.mean(losses,axis = 0)
    index = np.argmax(accuracy_per_epoch)
    best_accuracy = accuracy_per_epoch[index]
    best_epochs = index+1
    best_loss = loss_per_epoch[index]
    print(f"The training of hiperparams {config} has finalized, with loss: {best_loss}, out of fold accuracy: {best_accuracy}, epochs:{best_epochs}")
    return best_loss,best_accuracy,best_epochs
    #session.report({"loss": best_loss, "accuracy": best_accuracy,"epochs":best_epochs+1})


def get_best_model(model_class,epochs,config, train_obj,test_loader,path,is_kan ):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42) #Lets set a seed for the weights initialization
    if is_kan:
        model = model_class(grid_size = config["grid_size"])
    else:
        model = model_class()
 
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], eps=config["eps"],weight_decay = config["weight_decay"])
    
    train_loader = torch.utils.data.DataLoader(
        train_obj,
        batch_size=config["batch_size"],
        shuffle=True)
    
    all_train_loss, all_test_loss, all_test_accuracy, all_test_precision, all_test_recall, all_test_f1 = train_and_test_models(model, device, train_loader, test_loader, optimizer, criterion, epochs=epochs, scheduler=None,path= path)
    
    best_epochs = np.argmin(all_test_accuracy)
    best_accuracy = all_test_accuracy[best_epochs]
    best_loss = all_test_loss[best_epochs]
    return best_accuracy,best_loss

def search_hiperparams_and_get_final_model(model_class,is_kan, train_obj, test_loader,max_epochs,path,search_grid_combinations = 10,grid= {
    "lr":[1e-5, 1e-4,5e-4 ,1e-3],
    "weight_decay": [0, 1e-5, 1e-4],
    "batch_size":[32, 64, 128 ],
    "grid_size": [10,15,20]
    },folds = 3  ):

    
    best_trial = tune_hipers(model_class, is_kan, train_obj,max_epochs = max_epochs, n_combs = search_grid_combinations, grid = grid,folds = folds)
    epochs = best_trial['epochs']

    get_best_model(model_class,epochs,best_trial, train_obj,test_loader,path,is_kan)


def select_hipers_randomly(grid, n_combs,seed = 42):
    np.random.seed(seed) #Lets set a seed for the weights initialization
    combinations = []
    for i in range(n_combs):
        combination= {}
        for hiperparam in grid:
            combination[hiperparam] = (np.random.choice(grid[hiperparam]))
        combinations.append(combination)
    return combinations
