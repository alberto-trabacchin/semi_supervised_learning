from sklearn.datasets import make_classification
from matplotlib import pyplot as plt


if __name__ == "__main__":
    X, y = make_classification(
        n_samples = 1000,
        n_features = 10,
        n_classes = 5,
        shuffle = True,
        random_state = 42,
        n_clusters_per_class = 1,
        n_informative = 10,
        n_redundant = 0,
        n_repeated = 0
    )