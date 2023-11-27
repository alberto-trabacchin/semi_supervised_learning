import data_setup, model_builder, engine, utils
import torch
import wandb
import argparse
import os
import matplotlib.pyplot as plt


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", 
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
    parser.add_argument("--n_classes",
                        type = int,
                        default = 3,
                        help = "Number of classes in dataset")
    parser.add_argument("--n_features",
                        type = int,
                        default = 5,
                        help = "Number of features in dataset")
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
                        default = 100, 
                        help = "Number of epochs to train the model for")
    parser.add_argument("--batch_size", 
                        type = int, 
                        default = 64, 
                        help = "Number of samples per batch")
    parser.add_argument("--lr", 
                        type = float, 
                        default = 0.1, 
                        help = "Learning rate for optimizer")
    parser.add_argument("--num_workers", 
                        type = int, 
                        default = 4, 
                        help = "Number of workers for DataLoader")
    parser.add_argument("--device", 
                        type = str, 
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
    torch.manual_seed(args.random_state)

    wandb.init(
        # set the wandb project where this run will be logged
        project="Multiclass Classifiers Semi-Supervised Learning",
        name = f"{args.model_name}",
        mode = args.track_wandb,
        
        # track hyperparameters and run metadata
        config = {
        "learning_rate": args.lr,
        "test_size": args.test_size,
        "label_size": args.label_size,
        "hidden_size": args.hidden_size,
        "n_samples": args.n_samples,
        "n_classes": args.n_classes,
        "n_features": args.n_features,
        "noise": args.noise,
        "architecture": f"{args.model_name}-{args.hidden_size}x{args.hidden_size}x1",
        "dataset": "Moon-Dataset-SKLearn",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": args.device,
        }
    )

    train_lab_dl, train_unlab_dl, test_dl = data_setup.create_dataloaders(
        n_samples = args.n_samples,
        n_features = args.n_features,
        n_classes = args.n_classes,
        test_size = args.test_size,
        label_size = args.label_size,
        shuffle = True,
        random_state = args.random_state,
        batch_size = args.batch_size,
        num_workers = args.num_workers
    )
    utils.print_data_sizes(
        train_lab_dl,
        train_unlab_dl,
        test_dl
    )

    teacher_model = model_builder.ModelV1(
        input_size = args.n_features, 
        hidden_size = args.hidden_size,
        output_size = args.n_classes
    )
    student_model = model_builder.ModelV1(
        input_size = args.n_features, 
        hidden_size = args.hidden_size,
        output_size = args.n_classes
    )

    teacher_results = engine.train(
        model = teacher_model,
        train_dataloader = train_lab_dl,
        test_dataloader = test_dl,
        loss_fn = torch.nn.CrossEntropyLoss(),
        optimizer = torch.optim.Adam(teacher_model.parameters(), lr = args.lr),
        device = args.device,
        epochs = args.epochs,
        verbose = args.verbose,
        model_name = args.model_name,
        description = "Teacher",
        pseudo_labels = False
    )

    engine.predict_pseudo_labels(
        model = teacher_model,
        data_loader = train_unlab_dl,
        device = args.device
    )

    student_results = engine.train(
        model = student_model,
        train_dataloader = train_unlab_dl,
        test_dataloader = test_dl,
        loss_fn = torch.nn.CrossEntropyLoss(),
        optimizer = torch.optim.Adam(student_model.parameters(), lr = args.lr),
        device = args.device,
        epochs = args.epochs,
        verbose = args.verbose,
        model_name = args.model_name,
        description = "Student",
        pseudo_labels = True
    )

    wandb.finish()