def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


from evaluations import train_and_test_models
import torch.nn as nn
import torch.optim as optim
import time
import torch
import os
from torch.utils.data import DataLoader

import numpy as np


def train_model_generic(
    model, train_ds, test_ds, device, epochs=15, path="drive/MyDrive/KANs/models"
):
    model.to(device)
    print("Params start", count_parameters(model))

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    start = time.perf_counter()
    gen = torch.Generator()
    gen.manual_seed(0)
    mnist_train, mnist_val = torch.utils.data.random_split(
        train_ds, [51000, 9000], generator=gen
    )
    # DataLoader
    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    val_loader = DataLoader(mnist_val, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

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
        val_loader,
        optimizer,
        criterion,
        epochs=epochs,
        scheduler=scheduler,
        path=None,
    )

    best_epochs = np.argmax(all_test_accuracy) + 1

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

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
        epochs=best_epochs,
        scheduler=scheduler,
        path=path,
        save_last=True,
    )
    total_time = time.perf_counter() - start

    model.training_time = total_time / 60 / epochs
    print("Total time (min)", total_time / 60)
    if not path is None:
        saving_path = os.path.join(path, model.name + ".pt")
        model = torch.load(saving_path, map_location=torch.device(device))
        model.train_losses = all_train_loss
        model.test_losses = all_test_loss
        torch.save(model, saving_path)
    print("Train loss", all_train_loss)
    print("Test loss", all_test_loss)

    # return all_train_loss, all_test_loss, all_test_accuracy, all_test_precision, all_test_recall, all_test_f1
