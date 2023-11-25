from prettytable import PrettyTable
import torch
from torch.utils.data import DataLoader
from typing import Dict, List
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import requests

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("3_classification/helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download")
else:
    print("Downloading helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary


def plot_classes_distribution(dataloader: DataLoader, class_names: List[str]) -> None:
    target_ids = []
    mapper = {i: class_name for i, class_name in enumerate(class_names)}
    for batch in dataloader:
        target_ids.extend(batch[1].tolist())
    df = pd.DataFrame(target_ids, columns = ["target"])
    df["target_name"] = df.map(lambda x: mapper[x])
    df.target_name = df.target_name.str.title()
    df.target_name.value_counts().plot(
        kind = "bar", grid = True, xlabel = "Classes", ylabel = "Count",
        title = "Classes Distribution", rot = 45)
    plt.tight_layout()


def accuracy(y_pred_logit, y_true_classes):
    y_pred_class = torch.round(torch.sigmoid(y_pred_logit))
    return (y_pred_class == y_true_classes).sum().item() / len(y_true_classes)


def make_table_train_results(
        model_name: str,
        epoch: int,
        train_loss: float,
        test_loss: float,
        train_acc: float,
        test_acc: float
) -> None:
    table = PrettyTable(["Model Name", "Epoch", "Train Loss", "Train Accuracy", "Test Loss", "Test Accuracy"])
    table.add_row([model_name,
                   epoch,
                   f"{train_loss :.4f}",
                   f"{100 * train_acc :.2f}%",
                   f"{test_loss :.4f}",
                   f"{100 * test_acc :.2f}%"])
    return table


def print_models_results(models_results: Dict[str, dict]) -> None:
    table = PrettyTable(["Model Name", "Train Loss", "Train Accuracy", "Test Loss", "Test Accuracy"])
    for model_name, model_result in models_results.items():
        table.add_row([model_name,
                       f"{model_result['train_loss'][-1] :.4f}",
                       f"{100 * model_result['train_acc'][-1] :.2f}%",
                       f"{model_result['test_loss'][-1] :.4f}",
                       f"{100 * model_result['test_acc'][-1] :.2f}%"])
    print(table)
    

def count_parameters(model: torch.nn.Module, verbose = True) -> int:
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    if verbose:
        print(table)
        print(f"Total trainable params: {total_params}")
    return total_params


def save_model(name: str, model: torch.nn.Module, model_dir: Path) -> None:
    model_dir.mkdir(parents = True, exist_ok = True)
    model_path = model_dir / f"{name}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")


def load_model(name: str, model: torch.nn.Module, model_dir: Path) -> torch.nn.Module:
    model_path = model_dir / f"{name}.pth"
    model.load_state_dict(torch.load(model_path))
    print(f"Loaded model from {model_path}")
    return model


def plot_losses(model_results: Dict[str, dict], save_path: Path = None) -> None:
    epochs = np.array(model_results["epoch"], dtype = np.int32)
    train_loss = np.array(model_results["train_loss"], dtype = np.float32)
    test_loss = np.array(model_results["test_loss"], dtype = np.float32)
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.plot(epochs, 100 * train_loss, label = "train loss")
    ax.plot(model_results["epoch"], 100 * test_loss, label = "test loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Losses")
    ax.legend()
    if save_path is not None:
        save_path.parent[0].mkdir(parents = True, exist_ok = True)
        fig.set_size_inches(10, 8)
        fig.set_dpi(1000)
        fig.savefig(save_path)


def plot_accuracies(model_results: Dict[str, dict], save_name: Path = None) -> None:
    epochs = np.array(model_results["epoch"], dtype = np.int32)
    train_acc = np.array(model_results["train_acc"], dtype = np.float32)
    test_acc = np.array(model_results["test_acc"], dtype = np.float32)
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.plot(epochs, 100 * train_acc, label = "train accuracy")
    ax.plot(epochs, 100 * test_acc, label = "test accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracies")
    ax.legend()
    if save_name is not None:
        fig.set_size_inches(10, 8)
        fig.set_dpi(1000)
        fig.savefig(save_name)


def plot_dec_bounds(
        dataloader: DataLoader,
        model: torch.nn.Module
) -> None:
    X, y = dataloader.dataset.tensors
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

if __name__ == "__main__":
    model_results = {
        "train_loss": range(10),
        "train_acc": range(10),
        "test_loss": range(10),
        "test_acc": range(10),
        "epoch": range(10)
    }
    plot_losses(model_results)
    plot_accuracies(model_results)
    plt.show()