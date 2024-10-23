from tqdm import tqdm
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt
def train(model, device, train_loader, optimizer, epoch, criterion):
    """
    Train the model for one epoch

    Args:
        model: the neural network model
        device: cuda or cpu
        train_loader: DataLoader for training data
        optimizer: the optimizer to use (e.g. SGD)
        epoch: the current epoch
        criterion: the loss function (e.g. CrossEntropy)

    Returns:
        avg_loss: the average loss over the training set
    """

    model.to(device)
    model.train()
    train_loss = 0
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        # Recall that GPU is optimized for the operations we are dealing with
        data, target = data.to(device), target.to(device)

        # Reset the optimizer
        optimizer.zero_grad()

        # Push the data forward through the model layers
        output = model(data)

        # Get the loss
        loss = criterion(output, target) + model.reg_loss()

        # Keep a running total
        train_loss += loss.item()

        # Backpropagate
        loss.backward()
        optimizer.step()

    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx+1)
    # print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss

def test(model, device, test_loader, criterion):
    """
    Test the model

    Args:
        model: the neural network model
        device: cuda or cpu
        test_loader: DataLoader for test data
        criterion: the loss function (e.g. CrossEntropy)

    Returns:
        test_loss: the average loss over the test set
        accuracy: the accuracy of the model on the test set
        precision: the precision of the model on the test set
        recall: the recall of the model on the test set
        f1: the f1 score of the model on the test set
    """

    model.eval()
    test_loss = 0
    correct = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Get the predicted classes for this batch
            output = model(data)

            # Calculate the loss for this batch
            test_loss += criterion(output, target).item()

            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += (target == predicted).sum().item()

            # Collect all targets and predictions for metric calculations
            all_targets.extend(target.view_as(predicted).cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate overall metrics
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')

    # Normalize test loss
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Precision: {:.2f}, Recall: {:.2f}, F1 Score: {:.2f}\n'.format(
    #     test_loss, correct, len(test_loader.dataset), accuracy, precision, recall, f1))

    return test_loss, accuracy, precision, recall, f1
def train_and_test_models(model, device, train_loader, test_loader, optimizer, criterion, epochs, scheduler, path = "drive/MyDrive/KANs/models"):
    """
    Train and test the model

    Args:
        model: the neural network model
        device: cuda or cpu
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        optimizer: the optimizer to use (e.g. SGD)
        criterion: the loss function (e.g. CrossEntropy)
        epochs: the number of epochs to train
        scheduler: the learning rate scheduler

    Returns:
        all_train_loss: a list of the average training loss for each epoch
        all_test_loss: a list of the average test loss for each epoch
        all_test_accuracy: a list of the accuracy for each epoch
        all_test_precision: a list of the precision for each epoch
        all_test_recall: a list of the recall for each epoch
        all_test_f1: a list of the f1 score for each epoch
    """
    # Track metrics
    all_train_loss = []
    all_test_loss = []
    all_test_accuracy = []
    all_test_precision = []
    all_test_recall = []
    all_test_f1 = []
    best_acc = 0
    for epoch in range(1, epochs + 1):
        # Train the model
        train_loss = train(model, device, train_loader, optimizer, epoch, criterion)
        all_train_loss.append(train_loss)

        # Test the model
        test_loss, test_accuracy, test_precision, test_recall, test_f1 = test(model, device, test_loader, criterion)
        all_test_loss.append(test_loss)
        all_test_accuracy.append(test_accuracy)
        all_test_precision.append(test_precision)
        all_test_recall.append(test_recall)
        all_test_f1.append(test_f1)
        print(f'End of Epoch {epoch}: Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2%}')
        if test_accuracy>best_acc:
            best_acc = test_accuracy
            torch.save(model,os.path.join(path,model.name+".pt"))
        scheduler.step()
    model.all_test_accuracy = all_test_accuracy
    model.all_test_precision = all_test_precision
    model.all_test_f1 = all_test_f1
    model.all_test_recall = all_test_recall
    print("Best test accuracy", best_acc)
    return all_train_loss, all_test_loss, all_test_accuracy, all_test_precision, all_test_recall, all_test_f1

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def highlight_max(s):
    is_max = s == s.max()
    return ['font-weight: bold' if v else '' for v in is_max]
import numpy as np
import pandas as pd
def final_plots(models,test_loader,criterion,device,use_time = False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    accs = []
    precisions = []
    recalls = []
    f1s = []
    params_counts = []
    times = []
    for model in models:
        test_loss, accuracy, precision, recall, f1 = test(model, device, test_loader, criterion)
        ax1.plot(model.test_lossses, label=model.name)
        ax2.scatter(count_parameters(model),accuracy,  label=model.name)
        accs.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)  
        params_counts.append(count_parameters(model))
        if use_time :
            times.append(model.training_time)
        else:
            times.append(np.nan)
    ax1.set_title('Loss Test vs Epochs')    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    ax2.set_title('Number of Parameters vs Accuracy')
    ax2.set_xlabel('Number of Parameters')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend() 
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Listas para acumular datos

    # Creación del DataFrame
    df = pd.DataFrame({
        "Test Accuracy": accs,
        "Test Precision": precisions,
        "Test Recall": recalls,
        "Test F1 Score": f1s,
        "Number of Parameters": params_counts,
        "Time":times
    }, index=[m.name for m in models])

    df.to_csv('experiment_28x28.csv', index=False)

    # Aplicando el estilo
    df_styled = df.style.apply(highlight_max, subset=df.columns[:], axis=0).format('{:.3f}')
    return df_styled
from sklearn.metrics import RocCurveDisplay
def plot_roc_one_vs_rest_all_models(models, dataloader,n_classes,device):
    fig,axs = plt.subplots(len(models), figsize=(6, 6*n_classes))
    for m in range(len(models)):
        plot_roc_one_vs_rest(models[m],dataloader,n_classes,device,axs[m])
def plot_roc_one_vs_rest(model,dataloader,n_classes,device,ax):
    with torch.no_grad():
        preds = []
        model.eval()
        targets = []
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            targets.append(target.cpu().numpy())
            # Get the predicted classes for this batch
            output = model(data)
            preds.append(output.cpu().data.numpy())
    predictions = np.concatenate(preds)
    targets = np.concatenate(targets)
    predictions = np.exp(predictions) #porque usamos log softmax
    for class_id in range(n_classes):
        RocCurveDisplay.from_predictions(
            targets == class_id,
            predictions[:,class_id],
            name=f"ROC curve for {class_id}",
            ax=ax,
        )
    ax.set_title(f'ROC OvR {model.name}')    
    ax.set_xlabel('FP Rate')
    ax.set_ylabel('TP Rate')


def train_and_test_regularized(model, device, train_loader, test_loader, optimizer, criterion, epochs, scheduler, path = "drive/MyDrive/KANs/models",reg_weight=1):
    """
    Train and test the model

    Args:
        model: the neural network model
        device: cuda or cpu
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        optimizer: the optimizer to use (e.g. SGD)
        criterion: the loss function (e.g. CrossEntropy)
        epochs: the number of epochs to train
        scheduler: the learning rate scheduler

    Returns:
        all_train_loss: a list of the average training loss for each epoch
        all_test_loss: a list of the average test loss for each epoch
        all_test_accuracy: a list of the accuracy for each epoch
        all_test_precision: a list of the precision for each epoch
        all_test_recall: a list of the recall for each epoch
        all_test_f1: a list of the f1 score for each epoch
    """
    # Track metrics
    all_train_loss = []
    all_test_loss = []
    all_test_accuracy = []
    all_test_precision = []
    all_test_recall = []
    all_test_f1 = []
    best_acc = 0
    for epoch in range(1, epochs + 1):
        # Train the model
        train_loss = train_kkan_regularized(model, device, train_loader, optimizer, epoch, criterion,reg_weight)
        all_train_loss.append(train_loss)

        # Test the model
        test_loss, test_accuracy, test_precision, test_recall, test_f1 = test(model, device, test_loader, criterion)
        all_test_loss.append(test_loss)
        all_test_accuracy.append(test_accuracy)
        all_test_precision.append(test_precision)
        all_test_recall.append(test_recall)
        all_test_f1.append(test_f1)
        print(f'End of Epoch {epoch}: Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2%}')
        if test_accuracy>best_acc:
            best_acc = test_accuracy
            torch.save(model,os.path.join(path,model.name+".pt"))
        scheduler.step()
    model.all_test_accuracy = all_test_accuracy
    model.all_test_precision = all_test_precision
    model.all_test_f1 = all_test_f1
    model.all_test_recall = all_test_recall
    print("Best test accuracy", best_acc)
    return all_train_loss, all_test_loss, all_test_accuracy, all_test_precision, all_test_recall, all_test_f1
def train_kkan_regularized(model, device, train_loader, optimizer, epoch, criterion,reg_weight = 1):
    """
    Train the model for one epoch

    Args:
        model: the neural network model
        device: cuda or cpu
        train_loader: DataLoader for training data
        optimizer: the optimizer to use (e.g. SGD)
        epoch: the current epoch
        criterion: the loss function (e.g. CrossEntropy)

    Returns:
        avg_loss: the average loss over the training set
    """

    model.to(device)
    model.train()
    train_loss = 0
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        # Recall that GPU is optimized for the operations we are dealing with
        data, target = data.to(device), target.to(device)

        # Reset the optimizer
        optimizer.zero_grad()

        # Push the data forward through the model layers
        output = model(data)

        # Get the loss
        loss = criterion(output, target)+ reg_weight * model.regularization_loss()

        # Keep a running total
        train_loss += loss.item()

        # Backpropagate
        loss.backward()
        optimizer.step()

    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx+1)
    # print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss