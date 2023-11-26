import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from typing import Tuple, Dict, List
from pathlib import Path
from tqdm.auto import tqdm
import wandb

import utils


def train_step(
        model: torch.nn.Module,
        data_loader: DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:

        (0.1112, 0.8743)
    """
    model.train()
    train_loss = 0
    train_accuracy = 0
    for i, batch_sample in enumerate(data_loader):
        X, y = batch_sample
        X, y = X.to(device), y.to(device)
        y_logits = model(X).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))
        loss = loss_fn(y_logits, y)
        train_loss += loss.item()
        train_accuracy += utils.accuracy(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(data_loader)
    train_accuracy /= len(data_loader)
    return train_loss, train_accuracy


def test_step(
        model: torch.nn.Module,
        data_loader: DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device,
) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:

        (0.0223, 0.8985)
    """
    model.eval()
    test_loss = 0
    test_accuracy = 0
    with torch.inference_mode():
        for batch_sample in data_loader:
            X, y = batch_sample
            X, y = X.to(device), y.to(device)
            y_logits = model(X).squeeze()
            y_pred = torch.round(torch.sigmoid(y_logits))
            loss = loss_fn(y_logits, y)
            test_loss += loss.item()
            test_accuracy += utils.accuracy(y_pred, y)
        test_loss /= len(data_loader)
        test_accuracy /= len(data_loader)
    return test_loss, test_accuracy


def train(
        model: torch.nn.Module,
        model_name: str,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        device: str,
        verbose: bool = True
) -> Dict[str, List[float]]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for 
        each epoch.
        In the form: {train_loss: [...],
                    train_acc: [...],
                    test_loss: [...],
                    test_acc: [...]} 
        For example if training for epochs=2: 
                    {train_loss: [2.0616, 1.0537],
                    train_acc: [0.3945, 0.3945],
                    test_loss: [1.2641, 1.5706],
                    test_acc: [0.3400, 0.2973]} 
    """
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "epoch": []
    }
    train_accuracy = 0
    test_accuracy = 0
    train_loss = 0
    test_loss = 0
    for epoch in tqdm(range(epochs)):
        model = model.to(device)
        train_loss, train_accuracy = train_step(
            model = model,
            data_loader = train_dataloader,
            loss_fn = loss_fn,
            optimizer = optimizer,
            device = device
        )
        test_loss, test_accuracy = test_step(
            model = model,
            data_loader = test_dataloader,
            loss_fn = loss_fn,
            device = device
        )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_accuracy)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_accuracy)
        results["epoch"].append(epoch + 1)
        if verbose:
            results_table = utils.make_table_train_results(
                model_name = model_name,
                epoch = epoch + 1,
                train_loss = train_loss,
                test_loss = test_loss,
                train_acc = train_accuracy,
                test_acc = test_accuracy
            )
            tqdm.write(results_table.get_string())
        wandb.log({f"Train Accuracy {model_name}": train_accuracy,
                   f"Train Loss {model_name}": train_loss,
                   f"Test Accuracy {model_name}": test_accuracy,
                   f"Test Loss {model_name}": test_loss})
    return results


def get_pseudo_labels(
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device
) -> torch.Tensor:
    """Predicts on a PyTorch model.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tensor of predictions from the model.
    """
    model.eval()
    predictions = []
    with torch.inference_mode():
        for i, batch_sample in enumerate(data_loader):
            X, _ = batch_sample
            X = X.to(device)
            y_logits = model(X)
            y_pred = torch.round(torch.sigmoid(y_logits))
            predictions.append(y_pred.to("cpu"))
    return predictions