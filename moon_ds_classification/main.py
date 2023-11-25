import data_setup, utils, model_builder, engine
import torch
from pathlib import Path
import os
from matplotlib import pyplot as plt
import argparse
import wandb


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_points", 
                        type = int, 
                        default = 1000, 
                        help = "Number of points to generate")
    parser.add_argument("--noise", 
                        type = float, 
                        default = 0.0, 
                        help = "Noise to add to data")
    parser.add_argument("--test_size", 
                        type = float, 
                        default = 0.2, 
                        help = "Ratio of test data")
    parser.add_argument("--label_size", 
                        type = float, 
                        default = 0.1, 
                        help = "Ratio of labeled data")
    parser.add_argument("--hidden_size", 
                        type = int, 
                        default = 32, 
                        help = "Size of hidden layer")
    parser.add_argument("--results_path", 
                        type = str, 
                        default = "results/", 
                        help = "Path to save results")
    parser.add_argument("--model_name", 
                        type = str, 
                        required = True, 
                        help = "Name of model to train")
    parser.add_argument("--epochs", 
                        type = int, 
                        default = 5, 
                        help = "Number of epochs to train the model for")
    parser.add_argument("--batch_size", 
                        type = int, 
                        default = 32, 
                        help = "Number of samples per batch")
    parser.add_argument("--lr", 
                        type = float, 
                        default = 0.1, 
                        help = "Learning rate for optimizer")
    parser.add_argument("--num_workers", 
                        type = int, 
                        default = os.cpu_count(), 
                        help = "Number of workers for DataLoader")
    parser.add_argument("--device", 
                        type = torch.device, 
                        default = "cuda" if torch.cuda.is_available() else "cpu", 
                        help = "Device to train model on")
    parser.add_argument("--verbose", 
                        type = bool, 
                        default = True, 
                        help = "Whether or not to print results")
    parser.add_argument("--track_wandb", 
                        type = str, 
                        default = "disabled", 
                        choices = ["online", "offline", "disabled"], 
                        help = "Whether or not to track results with wandb")
    parser.add_argument("--random_state", 
                        type = int, 
                        default = 42, 
                        help = "Random state for reproducibility")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    N_POINTS = args.n_points
    NOISE = args.noise
    TEST_SIZE = args.test_size
    LABEL_SIZE = args.label_size
    HIDDEN_SIZE = args.hidden_size
    RESULTS_PATH = args.results_path
    MODEL_NAME = args.model_name
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr
    NUM_WORKERS = args.num_workers
    DEVICE = args.device
    VERBOSE = args.verbose
    TRACK_ONLINE = args.track_online
    RANDOM_STATE = args.random_state
    torch.manual_seed(RANDOM_STATE)

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Moon Dataset Supervised Learning",
        name = MODEL_NAME,
        mode = "online" if TRACK_ONLINE else "disabled",
        
        # track hyperparameters and run metadata
        config = {
        "learning_rate": LR,
        "test_size": TEST_SIZE,
        "label_size": LABEL_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "n_points": N_POINTS,
        "noise": NOISE,
        "architecture": f"{MODEL_NAME}-{HIDDEN_SIZE}x{HIDDEN_SIZE}x1",
        "dataset": "Moon-Dataset-SKLearn",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "device": DEVICE
        }
    )

    # Get dataloaders
    train_lab_dataloader, train_unlab_dataloader, test_dataloader = data_setup.create_dataloaders(
        n_samples = N_POINTS,
        noise = NOISE,
        test_size = TEST_SIZE,
        label_size = LABEL_SIZE,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        shuffle = True,
        random_state = RANDOM_STATE
    )
    utils.print_data_sizes(
        train_lab_dataloader,
        train_unlab_dataloader,
        test_dataloader
    )

    # Create model, optimizer, and loss function
    model = model_builder.ModelV1(
        input_size = train_lab_dataloader.dataset.tensors[0].shape[1],
        hidden_size = HIDDEN_SIZE
    )
    optimizer = torch.optim.SGD(model.parameters(), lr = LR)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Train model
    results = engine.train(
        model = model,
        model_name = MODEL_NAME,
        train_dataloader = train_lab_dataloader,
        test_dataloader = test_dataloader,
        loss_fn = loss_fn,
        optimizer = optimizer,
        epochs = EPOCHS,
        device = DEVICE,
        verbose = VERBOSE,
        track_online = TRACK_ONLINE,
        pseudo_labels = None
    )

    # Plot results
    utils.plot_losses(results)
    utils.plot_accuracies(results)
    utils.plot_dec_bounds(train_lab_dataloader, model)
    plt.show()

    wandb.finish()