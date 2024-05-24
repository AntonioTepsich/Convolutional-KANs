def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
from evaluations import train_and_test_models
import torch.nn as nn
import torch.optim as optim
import time
def train_model_generic(model, train_loader, test_loader,device,epochs= 15,path =  "drive/MyDrive/KANs/models"):
    model.to(device)
    print("Params start",count_parameters(model))

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    criterion = nn.CrossEntropyLoss()
    start = time.perf_counter()
    all_train_loss, all_test_loss, all_test_accuracy, all_test_precision, all_test_recall, all_test_f1 = train_and_test_models(model, device, train_loader, test_loader, optimizer, criterion, epochs=epochs, scheduler=scheduler,path= path)
    model.train_losses = all_train_loss
    model.test_losses = all_test_loss
    total_time = time.perf_counter() -start
    print("Params End",count_parameters(model))
    model.training_time = total_time/60
    print("Total time (min)",total_time/60)
    #return all_train_loss, all_test_loss, all_test_accuracy, all_test_precision, all_test_recall, all_test_f1