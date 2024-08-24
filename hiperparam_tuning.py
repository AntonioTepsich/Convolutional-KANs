from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
from evaluations import train_and_test_models

def train_tune(config,model_class, is_kan,train_obj=None, val_loader=None,epochs = 20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42) #Lets set a seed for the weights initialization
    if is_kan:
        model = model_class(grid_size = config["grid_size"])
    else:
        model = model_class()
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"],weight_decay = config["weight_decay"])
    train_loader = torch.utils.data.DataLoader(
        train_obj,
        batch_size=config["batch_size"],
        shuffle=True)
    
    all_train_loss, all_test_loss, all_test_accuracy, all_test_precision, all_test_recall, all_test_f1 = train_and_test_models(model, device, train_loader, val_loader, optimizer, criterion, epochs=epochs, scheduler=scheduler,path= None)
    best_epochs = np.argmin(all_test_accuracy)
    best_accuracy = all_test_accuracy[best_epochs]
    best_loss = all_test_loss[best_epochs]
    session.report({"loss": best_loss, "accuracy": best_accuracy,"epochs":best_epochs+1})

def tune_lr_betas_eps_l2(model_class, is_kan,train_obj,val_obj, n_combs = 20, max_epochs= 20, grid = {
    "lr":  tune.choice([1e-5, 1e-4,5e-4 ,1e-3]),
    "weight_decay":   tune.choice([0, 1e-5, 1e-4]),
    "batch_size":   tune.choice([ 32, 64,128 ]),
    "grid_size":[10,15,20]
    }):

    # Define the scheduler and reporter
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=30,
        grace_period=1,
        reduction_factor=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_obj,
        batch_size=64,
        shuffle=False)
    
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"]
    )

    # Run the hyperparameter tuning
    result = tune.run(
        tune.with_parameters(train_tune, model_class = model_class,is_kan=is_kan, train_obj=train_obj , val_loader=val_loader,epochs = max_epochs ),
        resources_per_trial={"cpu": 12, "gpu": 1},
        config=grid,
        num_samples=n_combs,
        scheduler=scheduler,
        progress_reporter=reporter,

        verbose = 2
    )

    # Get the best trial
    best_trial = result.get_best_trial("accuracy", "max", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
    print(f"Best trial final number of epochs: {best_trial.last_result['epochs']}")
    return best_trial#best_trial.last_result['epochs'], best_trial.last_result['accuracy']

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

def search_hiperparams_and_get_final_model(model_class,is_kan, train_obj, valid_obj, test_loader,path,search_grid_combinations = 10 ):
    best_trial = tune_lr_betas_eps_l2(model_class, is_kan, train_obj,valid_obj, n_combs = search_grid_combinations, grid = {
    "lr": tune.choice([1e-5, 1e-4,5e-4 ,1e-3]),
    "weight_decay": tune.choice([0, 1e-5, 1e-4]),
    "batch_size": tune.choice([32, 64, 128 ]),
    "grid_size": [10,15,20]
    })
    epochs = best_trial.last_result['epochs']
    get_best_model(model_class,epochs,best_trial, train_obj,test_loader,path)