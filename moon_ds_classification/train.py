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
    torch.manual_seed(args.random_state)

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Moon Dataset Supervised Learning",
        name = f"{args.model_name}",
        mode = args.track_wandb,
        
        # track hyperparameters and run metadata
        config = {
        "learning_rate": args.lr,
        "test_size": args.test_size,
        "label_size": args.label_size,
        "hidden_size": args.hidden_size,
        "n_points": args.n_points,
        "noise": args.noise,
        "architecture": f"{args.model_name}-{args.hidden_size}x{args.hidden_size}x1",
        "dataset": "Moon-Dataset-SKLearn",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": args.device,
        }
    )

    # Get dataloaders
    train_lab_dataloader, train_unlab_dataloader, test_dataloader = data_setup.create_dataloaders(
        n_samples = args.n_points,
        noise = args.noise,
        test_size = args.test_size,
        label_size = args.label_size,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        shuffle = True,
        random_state = args.random_state
    )
    utils.print_data_sizes(
        train_lab_dataloader,
        train_unlab_dataloader,
        test_dataloader
    )

    # Create models, optimizers, and loss function
    teacher_model = model_builder.ModelV1(
        input_size = train_lab_dataloader.dataset.tensors[0].shape[1],
        hidden_size = args.hidden_size
    )
    student_model = model_builder.ModelV1(
        input_size = train_lab_dataloader.dataset.tensors[0].shape[1],
        hidden_size = args.hidden_size
    )
    teacher_optimizer = torch.optim.SGD(teacher_model.parameters(), lr = args.lr)
    student_optimizer = torch.optim.SGD(student_model.parameters(), lr = args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Train the teacher
    print("\nTraining teacher...")
    teacher_results = engine.train(
        model = teacher_model,
        model_name = args.model_name,
        description = "Teacher",
        train_dataloader = train_lab_dataloader,
        test_dataloader = test_dataloader,
        loss_fn = loss_fn,
        optimizer = teacher_optimizer,
        epochs = args.epochs,
        device = args.device,
        verbose = args.verbose,
        pseudo_labels = False
    )

    # Predict pseudo-labels and update the unlabeled dataloader
    engine.predict_pseudo_labels(
        model = teacher_model,
        data_loader = train_unlab_dataloader,
        device = args.device
    )

    # Train the student
    print("\nTraining student...")
    student_results = engine.train(
        model = student_model,
        model_name = args.model_name,
        description = "Student",
        train_dataloader = train_unlab_dataloader,
        test_dataloader = test_dataloader,
        loss_fn = loss_fn,
        optimizer = student_optimizer,
        epochs = args.epochs,
        device = args.device,
        verbose = args.verbose,
        pseudo_labels = True
    )

    # Plot results
    print("\nModels results:")
    utils.print_models_results(model_name = f"{args.model_name}_Teacher", 
                               model_results = teacher_results)
    utils.print_models_results(model_name = f"{args.model_name}_Student",
                               model_results = student_results)
    utils.plot_losses(teacher_results, title = "Teacher Loss")
    utils.plot_accuracies(teacher_results, title = "Teacher Accuracy")
    utils.plot_dec_bounds(train_lab_dataloader, 
                          teacher_model,
                          pseudo_labels = None,
                          title = "Teacher Train-Lab-Decision-Boundaries")
    utils.plot_dec_bounds(train_unlab_dataloader, 
                          teacher_model, 
                          pseudo_labels = True,
                          title = "Teacher Train-Unlab-Decision-Boundaries")
    utils.plot_dec_bounds(test_dataloader, 
                          teacher_model,
                          pseudo_labels = None,
                          title = "Teacher Test-Decision-Boundaries")
    utils.plot_losses(student_results, title = "Student Loss")
    utils.plot_accuracies(student_results, title = "Student Accuracy")
    utils.plot_dec_bounds(train_unlab_dataloader, 
                          student_model,
                          pseudo_labels = False,
                          title = "Student Train-Unlab-Decision-Boundaries(GT)")
    utils.plot_dec_bounds(train_unlab_dataloader, 
                          student_model,
                          pseudo_labels = True,
                          title = "Student Train-Unlab-Decision-Boundaries(PL)")
    utils.plot_dec_bounds(test_dataloader, 
                          teacher_model,
                          pseudo_labels = None,
                          title = "Student Test-Decision-Boundaries")
    plt.tight_layout()
    plt.show()

    wandb.finish()