from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import os


class PseudoDataset(Dataset):
    def __init__(self, data: torch.Tensor, targets: torch.Tensor):
        self.data = data
        self.targets = targets
        self.pseudo_targets = torch.empty_like(self.targets)

    def __getitem__(self, index):
        return (
            self.data[index],
            self.targets[index],
            self.pseudo_targets[index],
            index
        )

    def __len__(self):
        return self.data.size(0)


def create_dataloaders(
        n_samples: int,
        n_features: int,
        n_classes: int,
        label_size: float,
        test_size: float,
        batch_size: int,
        num_workers: int = os.cpu_count(),
        shuffle: bool = True,
        random_state: int = 42
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.
        shuffle: Whether or not to shuffle the data in the DataLoader.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
        Example usage:
        train_dataloader, test_dataloader, class_names = \
            = create_dataloaders(train_dir = path/to/train_dir,
                                test_dir = path/to/test_dir,
                                transform = some_transform,
                                batch_size = 32,
                                num_workers = 4)
    """
    # Create the training and testing datasets
    dataset = make_classification(
        n_samples = n_samples,
        n_features = n_features,
        n_informative = n_features-1,
        n_redundant = 0,
        n_repeated = 0,
        n_classes = n_classes,
        shuffle = True,
        random_state = random_state,
        n_clusters_per_class = 1
    )
    X_train, X_test, y_train, y_test = train_test_split(
        dataset[0], 
        dataset[1], 
        test_size = test_size,
        shuffle = True,
        random_state = random_state
    )
    X_train_unlab, X_train_lab, y_train_unlab, y_train_lab = train_test_split(
        X_train, 
        y_train, 
        test_size = label_size,
        shuffle = True,
        random_state = random_state
    )

    # Convert to PyTorch tensors
    X_train_unlab = torch.FloatTensor(X_train_unlab)
    X_train_lab = torch.FloatTensor(X_train_lab)
    X_test = torch.FloatTensor(X_test)
    y_train_unlab = torch.LongTensor(y_train_unlab)
    y_train_lab = torch.LongTensor(y_train_lab)
    y_test = torch.LongTensor(y_test)

    # Convert to PyTorch datasets 
    train_lab_dataset = TensorDataset(X_train_lab, y_train_lab)
    test_dataset = TensorDataset(X_test, y_test)
    train_unlab_dataset = PseudoDataset(X_train_unlab, y_train_unlab)
    
    # Create dataloaders from PyTorch datasets
    train_lab_dataloader = DataLoader(dataset = train_lab_dataset,
                                      batch_size = len(train_lab_dataset),
                                      shuffle = shuffle,
                                      num_workers = num_workers)
    train_unlab_dataloader = DataLoader(dataset = train_unlab_dataset,
                                        batch_size = len(train_unlab_dataset),
                                        shuffle = shuffle,
                                        num_workers = num_workers)
    test_dataloader = DataLoader(dataset = test_dataset,
                                 batch_size = len(test_dataset),
                                 shuffle = shuffle,
                                 num_workers = num_workers)
    return train_lab_dataloader, train_unlab_dataloader, test_dataloader