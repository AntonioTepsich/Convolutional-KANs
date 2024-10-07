def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
from evaluations import train_and_test_models
import torch.nn as nn
import torch.optim as optim
import time
import torch
import os
import numpy as np
def train_model_generic(model, train_loader, test_loader,device,epochs= 15,path =  "drive/MyDrive/KANs/models"):
    model.to(device)
    print("Params start",count_parameters(model))

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    criterion = nn.CrossEntropyLoss()
    

    start = time.perf_counter()
    gen = torch.Generator()
    gen.manual_seed(0)
    mnist_train, mnist_val = torch.utils.data.random_split(train_loader, [51000,9000])

    all_train_loss, all_test_loss, all_test_accuracy, all_test_precision, all_test_recall, all_test_f1 = train_and_test_models(model, device, mnist_train, mnist_val, optimizer, criterion, epochs=epochs, scheduler=scheduler,path= None)

    best_epochs = np.argmax(all_test_accuracy)+1
    all_train_loss, all_test_loss, all_test_accuracy, all_test_precision, all_test_recall, all_test_f1 = train_and_test_models(model, device, train_loader, test_loader, optimizer, criterion, epochs=best_epochs, scheduler=scheduler,path= path,save_last=True)
    total_time = time.perf_counter() - start

    model.training_time = total_time/60 /epochs
    print("Total time (min)",total_time/60)
    if not path is None:
        saving_path = os.path.join(path,model.name+".pt")
        model =  torch.load(saving_path, map_location=torch.device(device))
        model.train_losses = all_train_loss
        model.test_losses = all_test_loss
        torch.save(model,saving_path)
    print("Train loss",all_train_loss)
    print("Test loss",all_test_loss)

    #return all_train_loss, all_test_loss, all_test_accuracy, all_test_precision, all_test_recall, all_test_f1