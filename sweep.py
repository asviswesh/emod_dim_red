import wandb
from autoencoder_pipeline import prepare_data, train_model, ResidualAutoencoder


# Sweep configuration
sweep_config = {
    "method": "bayes",  # Choose 'bayes' for Bayesian optimization
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
        # map from ~4500 features to 1000 latent features
        "hidden_dims": {
            "values": [[4500, 3000, 2000], [3000, 1500], [2000, 1500], [2000]]
        },
        "latent_dim": {"values": [1000]},  # This is currently fixed
        "batch_size": {"values": [512, 1024, 2048]},
        "use_batch_norm": {"values": [True, False]},
        "use_residual": {"values": [True, False]},
        "activation_fn": {"values": ["GELU", "ReLU", "LeakyReLU"]},
    },
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="autoencoder_sweeps")


# Training function for the sweep
def sweep_train():
    # Explicitly initialize a new wandb run (optional but good for clarity)
    run = wandb.init()

    # Convert wandb.config to a regular dictionary
    sweep_config_dict = dict(wandb.config)

    # Fixed values that are constant across all sweeps
    fixed_config = {
        "filename": "data/combined_data.csv",
        "shuffle_train": True,
        "dropout_rate": 0.0,
        "max_epochs": 100,
        "log_plot_freq": 20,
        "log_plots": True,
        "log_grads": False,
        "early_stopping_patience": 20,
    }

    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(
        fixed_config["filename"],
        sweep_config_dict["batch_size"],
        fixed_config["shuffle_train"],
    )

    # Determine MSE vs BCE loss weights by counting number of features
    num_binary_features = train_loader.dataset.dataset.binary_data.shape[1]
    num_real_features = train_loader.dataset.dataset.real_data.shape[1]
    num_total_features = num_binary_features + num_real_features
    print(f"Input dimension: {num_total_features}")

    binary_weight = num_binary_features / num_total_features
    real_weight = 8.3 * num_real_features / num_total_features

    # Dynamic values computed during training
    dynamic_config = {
        "input_dim": num_total_features,
        "real_weight": real_weight,
        "binary_weight": binary_weight,
    }

    # Combine the fixed, sweep, and dynamic config dictionaries
    config = {**fixed_config, **sweep_config_dict, **dynamic_config}

    # Initialize the model with the combined config
    model = ResidualAutoencoder(**config)

    # Train the model
    train_model(model, train_loader, val_loader, test_loader, config)

    # Explicitly finish the wandb run (optional, helps ensure proper logging)
    run.finish()


# Run the sweep
wandb.agent(sweep_id, function=sweep_train, count=30)