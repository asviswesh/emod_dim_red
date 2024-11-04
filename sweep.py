#######Sweep for Colab and local####################################
####################################################################

#Create WandB sweep and run it

import wandb
import argparse
import json
import torch
import torch.nn as nn
import os
import sys
import gc


# Boolean conversion function,
# Needed because when sending boolean values as String via argv causes
# Colab to interpret the value to be always True irrespective of the actual value
def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "yes"}:
        return True
    elif value.lower() in {"false", "no"}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}")

# Check if running in Google Colab
in_colab = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


# Simulate command-line arguments for Colab
if in_colab:
    sys.argv = [
        "",  # Placeholder for script name
        "--sweep_id", "",
        "--device", "0",
        "--base_dir", "/content/drive/MyDrive/Research_Levine/diffusion_notebooks",
        "--project_name", "Rick_Predictions_500_300_100",
        "--sweep_run_count", "1",
        "--max_epochs", "61",
        "--latent_dim", "100",
        "--save_predictions", "True",
        "--test_only", "False",
        "--identity_run", "False",
        "--subset_data_30", "False",
        "--subset_data_1_percent", "False"
    ]

if not in_colab:
    # For running locally we need to import the following
    from autoencoder_pipeline import prepare_data, train_model, ResidualAutoencoder, post_process_predictions

# Define arguments with argparse
parser = argparse.ArgumentParser()
parser.add_argument("--sweep_id", type=str, default=None, help="ID of an existing sweep to use")
parser.add_argument("--device", type=int, default=0, help="Device ID")
parser.add_argument("--project_name", type=str, default="Autoencoder_Identity_Case")
parser.add_argument("--base_dir", type=str, default="data", help="Base directory for input/output")
parser.add_argument("--sweep_run_count", type=int, default=1, help="Number of times to run the sweep")
parser.add_argument("--max_epochs", type=int, default=1, help="Maximum number of epochs for training")
parser.add_argument("--latent_dim", type=int, default=500, help="Dimensionality of the latent space")
parser.add_argument("--save_predictions", type=str2bool, default=True, help="Whether to save predictions to a CSV file")
parser.add_argument("--test_only", type=str2bool, default=False, help="Save predictions only for the test dataset if True")
parser.add_argument("--identity_run", type=str2bool, default=True, help="Run for the identity case if True")
parser.add_argument("--subset_data_30", type=str2bool, default=True, help="Use 30 rows dataset for quick code check")
parser.add_argument("--subset_data_1_percent", type=str2bool, default=False, help="Use 1% dataset for quick code check")

# Parse arguments
args = parser.parse_args()  # reads args from command line locally and from sys.argv on Colab

# Access the parsed arguments
sweep_id = args.sweep_id
device = args.device
project_name = args.project_name
base_dir = args.base_dir
sweep_run_count = args.sweep_run_count
max_epochs = args.max_epochs
latent_dim = args.latent_dim
save_predictions = args.save_predictions
test_only = args.test_only
identity_run = args.identity_run
subset_data_30 = args.subset_data_30
subset_data_1_percent = args.subset_data_1_percent

# Print parsed arguments to confirm
print("Parsed arguments:")
print("sweep_id:", sweep_id)
print("device:", device)
print("project_name:", project_name)
print("base_dir:", base_dir)
print("sweep_run_count:", sweep_run_count)
print("max_epochs:", max_epochs)
print("latent_dim:", latent_dim)
print("save_predictions:", save_predictions)
print("test_only:", test_only)
print("identity_run:", identity_run)
print("subset_data_30:", subset_data_30)
print("subset_data_1_percent:", subset_data_1_percent)

# setup input and output directories and filenames

# Directory Structure:
# - Inputs: contains the input files.
# - Inputs/generated_test_data: stores the generated raw test data file.
# - Output/predictions: contains the JSON metadata file for prediction analysis.
# - Output/predictions/full_dataset: holds predictions for the full dataset, organized by subdirectories:
#     - /500: predictions for latent dimension 500.
#     - /300: predictions for latent dimension 300.
#     - /100: predictions for latent dimension 100.
#     - /identity: predictions for identity run.
# - Output/predictions/test_dataset: stores predictions for the test dataset, with the same subdirectory pattern as full_dataset.

# Predictions File Naming Convention:
# Format: predicted_<full/test/identity_case>_dataset_<500/300/100>_latent_dim_input_<all_rows/30_rows/1_percent_rows>.csv
# Examples:
# - predicted_full_dataset_500_latent_dim_input_all_rows.csv
# - predicted_test_dataset_identity_case_100_latent_dim_input_30_rows.csv


input_dir = f"{base_dir}/Inputs"
test_dataset_raw_dir= None


if subset_data_30:
    input_file = f"{input_dir}/combined_data_subset_30_rows.csv"
    output_file_postfix = "input_30_rows"
elif subset_data_1_percent:
    input_file = f"{input_dir}/combined_data_subset_1_percent.csv"
    output_file_postfix = "input_1_percent_rows"
else:
    input_file = f"{input_dir}/combined_data.csv"
    output_file_postfix = "input_all_rows"

output_dir = f"{base_dir}/Output/predictions"

#Store information to be used for running tests to analyze the model performance
evaluation_metadata_filename = f"{output_dir}/evaluation_metadata.json"


if test_only:
    output_test_dataset_dir = f"{output_dir}/test_dataset"
    test_dataset_raw_dir= f"{input_dir}/generated_test_data"
    if identity_run:
        output_file = f"{output_test_dataset_dir}/identity/predicted_test_dataset_identity_case_{latent_dim}_latent_dim_{output_file_postfix}.csv"
    else:
        output_file = f"{output_test_dataset_dir}/{latent_dim}/predicted_test_dataset_{latent_dim}_latent_dim_{output_file_postfix}.csv"
else:
    output_full_dataset_dir = f"{output_dir}/full_dataset"
    if identity_run:
        output_file = f"{output_full_dataset_dir}/identity/predicted_full_dataset_identity_case_{latent_dim}_latent_dim_{output_file_postfix}.csv"
    else:
        output_file = f"{output_full_dataset_dir}/{latent_dim}/predicted_full_dataset_{latent_dim}_latent_dim_{output_file_postfix}.csv"



print("Input file path:", input_file)
print("Output file path:", output_file)

activation_fn_map = {
    "GELU": nn.GELU,
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "PReLU": nn.PReLU,
    "SELU": nn.SELU
}


# Training function for the sweep
def sweep_train():

    # Explicitly initialize a new wandb run (optional but good for clarity)
    run = wandb.init(project=project_name)

    # Convert wandb.config to a regular dictionary
    sweep_config_dict = dict(wandb.config)

    # Fixed values that are constant across all sweeps
    fixed_config = {
        "filename": f"{input_file}",
        "shuffle_train": True,
        "log_plot_freq": 30,
        "log_plots": True,
        "log_grads": False,
        "devices": [device],
    }


    # Annika: Modified to prepare data for Rick Final
    train_loader, val_loader, test_loader, scaler, col_list_dict, test_eir_suid, full_loader, all_eir_suid = prepare_data(
        fixed_config["filename"],
        sweep_config_dict["batch_size"],
        fixed_config["shuffle_train"],
        test_dataset_predictions=test_only,
        test_dataset_raw_dir=test_dataset_raw_dir,
        input_type=output_file_postfix
    )


    # Determine MSE vs BCE loss weights by counting number of features
    num_binary_features = train_loader.dataset.dataset.binary_data.shape[1]
    num_real_features = train_loader.dataset.dataset.real_data.shape[1]
    num_total_features = num_binary_features + num_real_features
    print(f"Input dimension: {num_total_features}")

    binary_weight = num_binary_features / num_total_features
    real_weight = 8.3 * num_real_features / num_total_features
    #real_weight = 2 * num_real_features / num_total_features #ANNIKA Testing, giving more weight to binary loss because binary data needs more training

    # Dynamic values computed during training
    if identity_run:
        dynamic_config = {
            "input_dim": num_total_features,
            "real_weight": real_weight,
            "binary_weight": binary_weight,
            "identity_if_no_compression": True,
            "hidden_dims": [num_total_features],
            "latent_dim": num_total_features
        }
    else:
        dynamic_config = {
            "input_dim": num_total_features,
            "real_weight": real_weight,
            "binary_weight": binary_weight,
            #rest of the configs are in sweep_config
        }


    # Combine the fixed, sweep, and dynamic config dictionaries
    config = {**fixed_config, **sweep_config_dict, **dynamic_config}

    # Get the activation function from the sweep config
    activation_fn = activation_fn_map[config["activation_fn"]]  # Map string to function

    # Add activation_fn to config
    config["activation_fn"] = activation_fn


    # Initialize the model with the combined config
    model = ResidualAutoencoder(**config)

    # Train the model
    train_model(model, train_loader, val_loader, test_loader, config)

    model.eval()  # Set the model to evaluation mode

    if test_only:

      # Predict on test data
      print("Predicting test data")

      #Generate predictions from the model's test results
      x_hat_binary = model.aggregated_test_results['x_hat_binary']
      x_hat_real = model.aggregated_test_results['x_hat_real']

      # Post-process the predictions
      final_predictions_df = post_process_predictions(
          x_hat_binary,
          x_hat_real,
          scaler,
          col_list_dict,
          test_eir_suid)

      if save_predictions:
          # Save the final post-processed predictions as a CSV file
          final_predictions_df.to_csv(f'{output_file}', index=False)

    else:

        # Predict on full data Rick Final
        print("Predicting full data")
        all_binary_preds, all_real_preds = [], []
        with torch.no_grad():
            for batch in full_loader:
                x_binary, x_real, _, _ = batch  # Unpack batch to get binary and real components
                binary_pred, real_pred = model(x_binary, x_real)
                all_binary_preds.append(binary_pred)
                all_real_preds.append(real_pred)

        all_binary_preds = torch.cat(all_binary_preds, dim=0)
        all_real_preds = torch.cat(all_real_preds, dim=0)

        # Process predictions
        final_predictions_df = post_process_predictions(
            all_binary_preds,
            all_real_preds,
            scaler,
            col_list_dict,
            all_eir_suid
        )

        if save_predictions:
            # Save the final post-processed predictions as a CSV file
            final_predictions_df.to_csv(f'{output_file}', index=False)


    # Code to save col_list_dict to a JSON file so that we can access it later in tests
    subset_col_list_dict = {
        "degenerate_cols": list(col_list_dict["degenerate_cols"].keys()),
        "binary_cols": col_list_dict["binary_cols"],
        "real_cols": col_list_dict["real_cols"],
        "original_cols": col_list_dict["original_cols"].tolist(),
        "integer_encoded_binary_cols":col_list_dict["integer_encoded_binary_cols"]
    }
    # Write to JSON file
    with open(evaluation_metadata_filename, "w") as f:
        json.dump(subset_col_list_dict, f)


    # Explicitly finish the wandb run (optional, helps ensure proper logging)
    run.finish()


    # Memory cleanup after training
    del model  # Delete the model to free memory
    del train_loader, val_loader, test_loader  # Free up memory

    if in_colab:
        torch.cuda.empty_cache()  # Clear GPU memory
        gc.collect()  # Force garbage collection



#Define the sweep configuration
if identity_run:
    sweep_config = {
        "method": "bayes",  # Bayesian optimization
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {
                "value": 6e-5
            },
            "batch_size": {"values": [2048]},
            "use_batch_norm": {"values": [True]},
            "use_residual": {"values": [False]},
            "activation_fn": {"values": ["PReLU"]},
            "gradient_clip_val": {"values": [0.0]},
            "dropout_rate": {"values": [0]},
            "max_epochs": {"values": [max_epochs]},
            "early_stopping_patience": {"values": [10]},
        }
    }

else:
    sweep_config = {
        "method": "bayes",  # Bayesian optimization
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {
                "value": 6e-5
            },
            # If you want to vary the learning rate uncomment this and comment the fixed rate above
            # "learning_rate": {
            #     "distribution": "log_uniform_values",
            #     "min": 1e-5,
            #     "max": 1e-4
            # },
            # Large architecture
            "hidden_dims": {
                "values": [
                    [12000, 8000, 4000, 2000, 1000]
                ]
            },
            "latent_dim": {"values": [latent_dim]},
            "batch_size": {"values": [2048]}, #smaller batch size
            "use_batch_norm": {"values": [True]},
            "use_residual": {"values": [False]},
            "activation_fn": {"values": ["PReLU"]},
            "gradient_clip_val": {"values": [0.0]},
            "dropout_rate": {"values": [0]},
            "max_epochs": {"values": [max_epochs]},
            "early_stopping_patience": {"values": [10]}

        },
        # Use this if you want early stopping strategy to limit tuner iterations
        # "early_terminate": {
        #     "type": "hyperband",
        #     "min_iter": 10
        # }
    }




# Initialize the sweep
if sweep_id:
    sweep_id = args.sweep_id
else:
    print("Creating new sweep")
    sweep_id = wandb.sweep(sweep_config, project=args.project_name)


# Run the sweep, runs count times
wandb.agent(sweep_id, function=sweep_train, count=sweep_run_count, project=project_name)
