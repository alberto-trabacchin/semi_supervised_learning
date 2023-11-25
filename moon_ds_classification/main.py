import data_setup, utils, model_builder, engine
import torch
from matplotlib import pyplot as plt
from helper_functions import plot_decision_boundary

if __name__ == "__main__":
    RANDOM_STATE = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LR = 0.001
    EPOCHS = 100
    N_POINTS = 1000
    NOISE = 0
    VERBOSE = True
    torch.manual_seed(RANDOM_STATE)

    # Get dataloaders
    train_lab_dataloader, train_unlab_dataloader, test_dataloader = data_setup.create_dataloaders(
        n_samples = N_POINTS,
        noise = NOISE,
        test_size = 0.2,
        label_size = 0.3,
        batch_size = 32,
        num_workers = 4,
        shuffle = True,
        random_state = RANDOM_STATE
    )

    # Create model, optimizer, and loss function
    model = model_builder.ModelV1(
        input_size = 2,
        hidden_size = 32
    )
    optimizer = torch.optim.SGD(model.parameters(), lr = LR)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Train model
    results = engine.train(
        model = model,
        model_name = "ModelV1",
        train_dataloader = train_lab_dataloader,
        test_dataloader = test_dataloader,
        loss_fn = loss_fn,
        optimizer = optimizer,
        epochs = EPOCHS,
        device = DEVICE,
        verbose = VERBOSE,
        track_online = False,
        pseudo_labels = None
    )

    # Plot results
    utils.plot_losses(results, save_path = None)
    utils.plot_accuracies(results, save_name = None)
    utils.plot_dec_bounds(
        dataloader= train_lab_dataloader,
        model = model
    )
    plt.show()