import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
from evaluations import train_and_test_models
from sklearn.model_selection import StratifiedKFold
import time


def tune_hipers(
    model_class,
    is_kan,
    train_obj,
    max_epochs,
    n_combs,
    grid,
    folds=3,
    save_file=True,
    dataset_name="MNIST",
    grid_size=20,
):
    combinations = select_hipers_randomly(grid, n_combs, seed=42)
    best_trial = {"accuracy": 0}
    if is_kan:
        nombre_modelo = model_class(grid_size=grid_size).name
    else:
        nombre_modelo = model_class().name
    for comb in combinations:
        start = time.perf_counter()
        loss, accuracy, epochs = train_tune(
            comb,
            model_class,
            is_kan,
            grid_size,
            train_obj=train_obj,
            epochs=max_epochs,
            folds=folds,
        )
        if best_trial["accuracy"] < accuracy:
            best_trial["accuracy"] = accuracy
            best_trial["epochs"] = epochs
            best_trial["loss"] = loss
            best_trial.update(comb)

        if save_file:
            with open(f"results/{dataset_name}/{nombre_modelo} logs.txt", "a+") as f:
                f.write(
                    f"Finished Trial with Hipers {comb} and got accuracy {accuracy} with epochs {epochs}. Took {time.perf_counter()- start} seconds \n"
                )

        print(
            f"Finished Trial with Hipers {comb} and got accuracy {accuracy} with epochs {epochs}"
        )
    # Get the best trial
    print(f"Best trial config: {best_trial}")
    print(f"Best trial final validation loss: {best_trial['loss']}")
    print(f"Best trial final validation accuracy: {best_trial['accuracy']}")
    print(f"Best trial final number of epochs: {best_trial['epochs']}")
    if save_file:
        with open(f"results/{dataset_name}/{nombre_modelo} final result.txt", "w") as f:
            f.writelines(
                [
                    f"Best trial config: {best_trial}",
                    f"Best trial final validation loss: {best_trial['loss']}",
                    f"Best trial final validation accuracy: {best_trial['accuracy']}",
                    f"Best trial final number of epochs: {best_trial['epochs']}",
                ]
            )

    return best_trial  # best_trial.last_result['epochs'], best_trial.last_result['accuracy']


class TrainValSplit:
    def __init__(
        self,
        dataset_object,
        train_pctg=0.8,
    ):
        self.train_pctg = train_pctg
        gen = torch.Generator()
        gen.manual_seed(0)
        train_samples = int(len(dataset_object) * train_pctg)
        val_samples = len(dataset_object) - train_samples
        self.train, self.valid = torch.utils.data.random_split(
            dataset_object, [train_samples, val_samples], generator=gen
        )

    def split(self, _, _2):
        return [(self.train, self.valid)]


class kfoldsplit:
    def __init__(self, train_obj, n_splits=3, shuffle=True, random_state=1):
        self.train_obj = train_obj
        self.kfold = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def split(self, indexes, targets):
        train_ids, valid_ids = self.kfold.split(indexes, targets)

        trainset = torch.utils.data.Subset(self.train_obj, train_ids)
        validset = torch.utils.data.Subset(self.train_obj, valid_ids)
        # Define data loaders for training and testing data in this fold
        return [(trainset, validset)]


def train_tune(
    config, model_class, is_kan, grid_size, train_obj=None, epochs=20, folds=3
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)  # Lets set a seed for the weights initialization
    if folds > 1:
        splitter = kfoldsplit(
            train_obj
        )  # StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)
    else:
        splitter = TrainValSplit(train_obj)
    accuracys = []
    losses = []
    # print(config)
    for trainset, validset in splitter.split(
        np.arange(len(train_obj)), train_obj.targets
    ):
        # print("starting fold", fold)
        if is_kan:
            model = model_class(grid_size=grid_size)
        else:
            model = model_class()
        # Sample elements randomly from a given list of ids, no replacement.
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=int(config["batch_size"])
        )
        valid_loader = torch.utils.data.DataLoader(
            validset, batch_size=int(config["batch_size"])
        )
        # Init the neural network
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        optimizer = optim.AdamW(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

        (
            all_train_loss,
            all_test_loss,
            all_test_accuracy,
            all_test_precision,
            all_test_recall,
            all_test_f1,
        ) = train_and_test_models(
            model,
            device,
            train_loader,
            valid_loader,
            optimizer,
            criterion,
            epochs=epochs,
            scheduler=None,
            path=None,
            verbose=True,
        )
        accuracys.append(all_test_accuracy)
        losses.append(all_test_loss)

    accuracy_per_epoch = np.mean(accuracys, axis=0)

    loss_per_epoch = np.mean(losses, axis=0)
    index = np.argmax(accuracy_per_epoch)
    best_accuracy = accuracy_per_epoch[index]
    best_epochs = index + 1
    best_loss = loss_per_epoch[index]
    print(
        f"The training of hiperparams {config} has finalized, with loss: {best_loss}, out of fold accuracy: {best_accuracy}, epochs:{best_epochs}"
    )
    return best_loss, best_accuracy, best_epochs


def get_best_model(
    model_class, epochs, config, train_obj, test_loader, path, is_kan, grid_size
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)  # Lets set a seed for the weights initialization
    if is_kan:
        model = model_class(grid_size=grid_size)
    else:
        model = model_class()

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    train_loader = torch.utils.data.DataLoader(
        train_obj, batch_size=int(config["batch_size"]), shuffle=True
    )

    (
        all_train_loss,
        all_test_loss,
        all_test_accuracy,
        all_test_precision,
        all_test_recall,
        all_test_f1,
    ) = train_and_test_models(
        model,
        device,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        epochs=epochs,
        scheduler=None,
        path=path,
        patience=6,
    )

    best_epochs = np.argmax(all_test_accuracy)
    best_accuracy = all_test_accuracy[best_epochs]
    best_loss = all_test_loss[best_epochs]
    return best_accuracy, best_loss


def search_hiperparams_and_get_final_model(
    model_class,
    is_kan,
    train_obj,
    test_loader,
    max_epochs,
    path,
    search_grid_combinations=10,
    grid={
        "lr": [1e-4, 5e-4, 1e-3],
        "weight_decay": [0, 1e-5, 1e-4],
        "batch_size": [32, 64, 128],
    },
    grid_size=20,
    folds=3,
    dataset_name="MNIST",
):

    best_trial = tune_hipers(
        model_class,
        is_kan,
        train_obj,
        max_epochs=max_epochs,
        n_combs=search_grid_combinations,
        grid=grid,
        folds=folds,
        dataset_name=dataset_name,
        grid_size=grid_size,
    )
    epochs = best_trial["epochs"]

    get_best_model(
        model_class, epochs, best_trial, train_obj, test_loader, path, is_kan, grid_size
    )


def select_hipers_randomly(grid, n_combs, seed=42, special_comb=False):
    np.random.seed(seed)  # Lets set a seed for the weights initialization
    combinations = []
    for i in range(n_combs):
        combination = {}
        for hiperparam in grid:
            combination[hiperparam] = np.random.choice(grid[hiperparam])
        combinations.append(combination)
    if special_comb:
        special_comb = {"lr": 0.0005, "weight_decay": 1e-05, "batch_size": 128}
        if not special_comb in combinations:
            combinations.append(
                special_comb
            )  # this seems to be one of the best and we dont want to leave it out to the random wheter its in or not
    return combinations
