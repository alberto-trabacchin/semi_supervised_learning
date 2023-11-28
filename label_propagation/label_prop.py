from typing import Dict
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation
import numpy as np
from matplotlib import pyplot as plt

import utils


def run_experiment(
        n_features: int,
        n_classes: int,
        n_samples: int,
        test_size: float,
        label_size: float,
        random_state: int = 42,
        max_iter: int = int(10e3)
) -> Dict[str, float]:
    
    # Generate data
    X, y = make_classification(
        n_samples,
        n_features,
        n_informative = n_features,
        n_redundant = 0,
        n_repeated = 0,
        n_classes = n_classes,
        shuffle = True,
        random_state = random_state,
        n_clusters_per_class = 1,
        flip_y = 0
    )

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size = test_size,
        shuffle = True,
        random_state = random_state
    )

    # Generate unlabeled data
    unlabeled_size = int((1 - label_size) * len(y_train))
    unlabeled_idx = np.arange(unlabeled_size)
    labels = np.copy(y_train)
    labels[unlabeled_idx] = -1

    # Fit label propagation model
    lab_prop_model = LabelPropagation(max_iter = max_iter)
    lab_prop_model.fit(X_train, labels)
    score = lab_prop_model.score(X_test, y_test)

    utils.plot2d_dataset(X = X_train, y = labels,
                         legend = ["Unlabeled", "Class A", "Class B", "Class C", "Class D"],
                         title = "Original dataset")
    plt.gcf().savefig("/home/alberto/Downloads/original_dataset.png", dpi = 600)
    labels[unlabeled_idx] = lab_prop_model.predict(X_train[unlabeled_idx])
    utils.plot2d_dataset(X = X_train, y = labels,
                         legend = ["Class A", "Class B", "Class C", "Class D"],
                         title = "Dataset with predicted labels")
    plt.gcf().savefig("/home/alberto/Downloads/predicted_dataset.png", dpi = 600)
    utils.plot2d_dataset(X = X_train, y = y_train,
                         legend = ["Class A", "Class B", "Class C", "Class D"],
                         title = "Dataset with true labels")
    plt.gcf().savefig("/home/alberto/Downloads/true_dataset.png", dpi = 600)
    return score