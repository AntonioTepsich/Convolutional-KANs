
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
from evaluations import train_and_test_models

def tune_hipers(model_class, is_kan, train_obj,valid_obj, n_combs , grid ):
    combinations = select_hipers_randomly(grid, n_combs,seed = 42)
    best_trial = {"accuracy": 0}
    for comb in combinations:
        loss,accuracy,epochs = train_tune(comb,model_class, is_kan,train_obj=train_obj, val_loader=valid_obj,epochs = 20)
        if best_trial.accuracy<accuracy:
            best_trial["accuracy"] = accuracy
            best_trial["epochs"] = epochs
            best_trial["loss"] = loss
        print(f"Finished Trial with Hipers {comb} and got accuracy {accuracy} with epochs {epochs}")
    # Get the best trial
    #best_trial = result.get_best_trial("accuracy", "max", "last")
    print(f"Best trial config: {best_trial}")
    print(f"Best trial final validation loss: {best_trial['loss']}")
    print(f"Best trial final validation accuracy: {best_trial['accuracy']}")
    print(f"Best trial final number of epochs: {best_trial['epochs']}")
    return best_trial#best_trial.last_result['epochs'], best_trial.last_result['accuracy']

def select_hipers_randomly(grid, n_combs,seed = 42):
    np.random.seed(seed) #Lets set a seed for the weights initialization
    combinations = []
    for i in range(n_combs):
        combination= {}
        for hiperparam in grid:
            combination[hiperparam] = (np.random.choice(grid[hiperparam]))
        combinations.append(combination)
    return combinations


def train_tune(config,model_class, is_kan,train_obj=None, val_loader=None,epochs = 20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42) #Lets set a seed for the weights initialization
    if is_kan:
        model = model_class(grid_size = config["grid_size"])
    else:
        model = model_class()
    print("config",config)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"],weight_decay = config["weight_decay"])
    train_loader = torch.utils.data.DataLoader(
        train_obj,
        batch_size=int(config["batch_size"]),
        shuffle=True)
    
    all_train_loss, all_test_loss, all_test_accuracy, all_test_precision, all_test_recall, all_test_f1 = train_and_test_models(model, device, train_loader, val_loader, optimizer, criterion, epochs=epochs, scheduler=None,path= None,verbose= False)
    best_epochs = np.argmin(all_test_accuracy)
    best_accuracy = all_test_accuracy[best_epochs]
    best_loss = all_test_loss[best_epochs]
    print(f"The training of hiperparams {config} has finalized, with loss: {best_loss}, accuracy: {best_accuracy},epochs:{best_epochs+1}")
    return best_loss,best_accuracy,best_epochs+1
    #session.report({"loss": best_loss, "accuracy": best_accuracy,"epochs":best_epochs+1})


def get_best_model(model_class,epochs,config, train_obj,test_loader,path,is_kan):
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

def search_hiperparams_and_get_final_model(model_class,is_kan, train_obj, valid_obj, test_loader,path,search_grid_combinations = 10,grid= {
    "lr":[1e-5, 1e-4,5e-4 ,1e-3],
    "weight_decay": [0, 1e-5, 1e-4],
    "batch_size":[32, 64, 128 ],
    "grid_size": [10,15,20]
    } ):
    val_loader = torch.utils.data.DataLoader(
        valid_obj,
        batch_size=128,
        shuffle=True)
    
    best_trial = tune_hipers(model_class, is_kan, train_obj,val_loader, n_combs = search_grid_combinations, grid = grid)
    epochs = best_trial['epochs']
    train_dev_sets = torch.utils.data.ConcatDataset([train_obj, train_obj])

    get_best_model(model_class,epochs,best_trial, train_dev_sets,test_loader,path)