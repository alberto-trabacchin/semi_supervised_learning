import label_prop, utils


if __name__ == "__main__":
    score = label_prop.run_experiment(
        n_features = 4,
        n_classes = 4,
        n_samples = 1000,
        test_size = 0.2,
        label_size = 0.01,
        random_state = 42,
        max_iter = int(10e4)
    )
    print(score)