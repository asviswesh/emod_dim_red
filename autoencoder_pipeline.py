# Autoencoder

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import roc_auc_score
from pdb import set_trace as bp
import json

# Check if running in Google Colab
in_colab = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ

if not in_colab:
    from utils.data_utils import *


class ResidualAutoencoder(pl.LightningModule):
    def __init__(self, input_dim, latent_dim, hidden_dims, activation_fn, use_residual,
                 identity_if_no_compression=False, use_batch_norm=False, use_layer_norm=False,
                 real_weight=1.0,
                 binary_weight=1.0,
                 dropout_rate=0.3,
                 learning_rate=1e-4,
                 max_epochs=10,
                 log_plot_freq=50,
                 log_plots=True,
                 log_grads=False,
                 **kwargs):
        super(ResidualAutoencoder, self).__init__()

        # print the kwargs and point out that they will be ignored
        print(f"kwargs: {kwargs}")
        print("Note: The kwargs will be ignored in this model, but were hopefully used elsewhere.")

        # set activation function if it is a string
        if isinstance(activation_fn, str):
             activation_fn = getattr(nn, activation_fn)

        self.log_plot_freq = log_plot_freq # how often to make plots and log them to wandb
        self.log_plots = log_plots
        self.log_grads = log_grads
        self.use_residual = use_residual
        self.identity_if_no_compression = identity_if_no_compression
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.real_weight = real_weight
        self.binary_weight = binary_weight

        self.test_outputs = []
        self.aggregated_test_results = {}


        # recommend to rescale MSE_loss to be of similar magnitude to BCE_loss
        # e.g. mse_loss_weight = 8.32 allows losses to be equal
        # also need to account for the number of features in each group
        # SO: total_loss = real_weight * MSE_loss + binary_weight * BCE_loss
        # where real_weight = 8.32 * num_real_features / num_total_features
        # and binary_weight = num_binary_features / num_total_features

        # Ensure that BatchNorm and LayerNorm are mutually exclusive
        if self.use_batch_norm and self.use_layer_norm:
            raise ValueError("You can only use one of BatchNorm or LayerNorm, not both.")

        self.return_identity = self.latent_dim == self.input_dim and self.identity_if_no_compression

        if self.return_identity:
            self.scalar_param = nn.Parameter(torch.ones(1))
        else:
            # Encoder with optional residual connections
            self.encoder_layers = nn.ModuleList()
            self.encoder_projections = nn.ModuleList()

            # First layer (input to first hidden)
            self.encoder_layers.append(self._create_layer(input_dim, hidden_dims[0], activation_fn))
            if self.use_residual:
                self.encoder_projections.append(
                    nn.Linear(input_dim, hidden_dims[0]) if input_dim != hidden_dims[0] else nn.Identity()
                )

            # Hidden layers
            for i in range(len(hidden_dims) - 1):
                self.encoder_layers.append(self._create_layer(hidden_dims[i], hidden_dims[i + 1], activation_fn))
                if self.use_residual:
                    self.encoder_projections.append(
                        nn.Linear(hidden_dims[i], hidden_dims[i + 1]) if hidden_dims[i] != hidden_dims[i + 1] else nn.Identity()
                    )

            # Latent layer
            self.latent_layer = nn.Linear(hidden_dims[-1], latent_dim)

            # Decoder with optional residual connections
            self.decoder_layers = nn.ModuleList()
            self.decoder_projections = nn.ModuleList()

            # First layer (latent to first hidden)
            self.decoder_layers.append(self._create_layer(latent_dim, hidden_dims[-1], activation_fn))
            if self.use_residual:
                self.decoder_projections.append(
                    nn.Linear(latent_dim, hidden_dims[-1]) if latent_dim != hidden_dims[-1] else nn.Identity()
                )

            # Hidden layers
            for i in range(len(hidden_dims) - 1, 0, -1):
                self.decoder_layers.append(self._create_layer(hidden_dims[i], hidden_dims[i - 1], activation_fn))
                if self.use_residual:
                    self.decoder_projections.append(
                        nn.Linear(hidden_dims[i], hidden_dims[i - 1]) if hidden_dims[i] != hidden_dims[i - 1] else nn.Identity()
                    )

            # Output layer
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dims[0], input_dim),
                nn.Sigmoid()  # Use Sigmoid since data is normalized between [0,1]
            )

    def _create_layer(self, in_dim, out_dim, activation_fn):
        """Helper method to create a single layer with optional BatchNorm or LayerNorm."""
        layers = [nn.Linear(in_dim, out_dim)]

        # Apply BatchNorm or LayerNorm based on the flags
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        elif self.use_layer_norm:
            layers.append(nn.LayerNorm(out_dim))

        layers.append(activation_fn())  # Add the activation function

        # Add dropout layer
        if self.dropout_rate > 0:
            layers.append(nn.Dropout(self.dropout_rate))

        return nn.Sequential(*layers)

    def forward(self, x_binary, x_real):
        # Concatenate binary and real data before encoding
        x = torch.cat([x_binary, x_real], dim=1)

        if self.return_identity:
            # Identity mapping case
            reconstructed = x * self.scalar_param  # Return input directly as reconstruction
        else:
            # Standard encoder-decoder with residuals
            x_encoded = self.encode(x)
            reconstructed = self.decode(x_encoded)

        # Split the reconstructed output back into binary and real parts
        x_hat_binary = reconstructed[:, :x_binary.shape[1]]
        x_hat_real = reconstructed[:, x_binary.shape[1]:]

        return x_hat_binary, x_hat_real

    def encode(self, x):
        # Pass through encoder layers
        for i, layer in enumerate(self.encoder_layers):
            identity = self.encoder_projections[i](x) if self.use_residual else 0
            x = layer(x) + identity
        return self.latent_layer(x)

    def decode(self, x):
        # Pass through decoder layers
        for i, layer in enumerate(self.decoder_layers):
            identity = self.decoder_projections[i](x) if self.use_residual else 0
            x = layer(x) + identity
        return self.output_layer(x)

    def training_step(self, batch, batch_idx):
        x_binary, x_real, mask_binary, mask_real = batch
        x_hat_binary, x_hat_real = self(x_binary, x_real)
        loss = self.compute_loss(x_binary, x_real,
                x_hat_binary, x_hat_real,
                mask_binary, mask_real,
                stage="train")

        # Log the true and reconstructed data as images to wandb
        if self.current_epoch % self.log_plot_freq == 0 and batch_idx == 0:  # Log only once per epoch
            # log binary images
            self.log_images_to_wandb(x_binary, x_hat_binary, x_real, x_hat_real, mask_binary, mask_real, 'train')


        return loss


    def validation_step(self, batch, batch_idx):
        x_binary, x_real, mask_binary, mask_real = batch
        x_hat_binary, x_hat_real = self(x_binary, x_real)
        loss = self.compute_loss(x_binary, x_real,
                x_hat_binary, x_hat_real,
                mask_binary, mask_real,
                stage="val")

        # Log the true and reconstructed data as images to wandb
        if (self.current_epoch % self.log_plot_freq == 0 and batch_idx == 0):
            # log binary images
            self.log_images_to_wandb(x_binary, x_hat_binary, x_real, x_hat_real, mask_binary, mask_real, 'val')



    def test_step(self, batch, batch_idx):
        x_binary, x_real, mask_binary, mask_real = batch
        x_hat_binary, x_hat_real = self(x_binary, x_real)
        loss = self.compute_loss(
            x_binary,
            x_real,
            x_hat_binary,
            x_hat_real,
            mask_binary,
            mask_real,
            stage="test",
        )

        # Log the true and reconstructed data as images to wandb
        if batch_idx == 0:
            # log binary images
            self.log_images_to_wandb(x_binary, x_hat_binary, x_real, x_hat_real, mask_binary, mask_real, "test")

        # Store predictions and masks in self.test_outputs
        output = {
            'x_hat_binary': x_hat_binary,
            'x_hat_real': x_hat_real
        }
        self.test_outputs.append(output)

    def on_test_epoch_end(self):
        # Aggregate the predictions and masks from each batch
        all_x_hat_binary = torch.cat([output['x_hat_binary'] for output in self.test_outputs], dim=0)
        all_x_hat_real = torch.cat([output['x_hat_real'] for output in self.test_outputs], dim=0)

        # Store the aggregated predictions
        self.aggregated_test_results = {
            'x_hat_binary': all_x_hat_binary,
            'x_hat_real': all_x_hat_real
        }

        # Clear test outputs to free up memory
        self.test_outputs = []

    def configure_optimizers(self):
        # Define optimizer
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        # Define the number of warmup steps
        num_warmup_steps = 250  # Can change this if required

        # Define the warmup schedule using LambdaLR
        def warmup_scheduler_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return 1.0  # Return the full learning rate after warmup period

        # LambdaLR for warmup
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_scheduler_lambda)

        # Return the optimizer and the warmup scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warmup_scheduler,  # Warmup scheduler
                "interval": "step",  # Apply the scheduler per step (batch)
                "frequency": 1,  # Apply it at every step
                "name": "warmup_lr_scheduler"  # Give a name for tracking in WandB
            },
        }

    def compute_loss(self, x_binary, x_real, x_hat_binary, x_hat_real, mask_binary, mask_real, stage):

        # MSE Loss for real-valued columns
        mse_loss = F.mse_loss(x_hat_real, x_real, reduction='none')
        mse_loss = (mse_loss * mask_real).mean()

        # Binary Cross-Entropy Loss for binary columns
        bce_loss = F.binary_cross_entropy(x_hat_binary, x_binary, reduction='none')
        bce_loss = (bce_loss * mask_binary).mean()

        weighted_mse_loss = self.real_weight * mse_loss
        weighted_bce_loss = self.binary_weight * bce_loss

        # Combine losses
        total_loss = weighted_mse_loss + weighted_bce_loss

        # a note that a random prediction (0.5) would have a BCE loss of 0.69
        # for real-valued data, unclear what the MSE loss would be for 0.5,
        # as it depends on the distribution of the data.
        # if the data are uniformly distributed between 0 and 1, then the MSE loss for 0.5
        # would be the variance of the distribution, which is 1/12 = 0.0833.
        # This would suggest a rescaling of:
        # total_loss = lambda * MSE_loss + BCE_loss
        # where lambda = 0.69 / 0.0833 = 8.32

        # Log losses to wandb
        self.log(f"{stage}_weighted_mse_loss", weighted_mse_loss.item(), on_epoch=True)
        self.log(f"{stage}_weighted_bce_loss", weighted_bce_loss.item(), on_epoch=True)
        self.log(f"{stage}_mse_loss", mse_loss.item(), on_epoch=True)
        self.log(f"{stage}_bce_loss", bce_loss.item(), on_epoch=True)
        self.log(f"{stage}_loss", total_loss.item(), on_epoch=True)

        # compute AUROC for binary columns
        # Compute AUROC only on valid data (non-missing values where mask == 1)
        x_binary_cpu = x_binary.detach().cpu().numpy()
        x_hat_binary_cpu = (
            x_hat_binary.detach().cpu().numpy()
        )  # No need for additional sigmoid
        mask_binary_cpu = mask_binary.detach().cpu().numpy()

        # Compute AUROC for each binary variable (mask out missing values)
        auroc_per_label = [
            roc_auc_score(
                x_binary_cpu[mask_binary_cpu[:, i] == 1, i],
                x_hat_binary_cpu[mask_binary_cpu[:, i] == 1, i],
            )
            for i in range(x_binary_cpu.shape[1])
            if len(np.unique(x_binary_cpu[mask_binary_cpu[:, i] == 1, i]))
            > 1  # Only compute if variance exists
        ]

        # Macro-Averaged AUROC
        if len(auroc_per_label) > 0:
            macro_auroc = np.mean(auroc_per_label)
        else:
            macro_auroc = float("nan")  # Handle cases where no AUROC could be computed

        # Flattened true and predicted arrays for micro-averaged AUROC
        y_true_flat = x_binary_cpu[mask_binary_cpu == 1].ravel()
        y_pred_flat = x_hat_binary_cpu[mask_binary_cpu == 1].ravel()

        if len(np.unique(y_true_flat)) > 1:  # Ensure there's variance in flattened data
            micro_auroc = roc_auc_score(y_true_flat, y_pred_flat)
        else:
            micro_auroc = float("nan")

        # Log AUROCs to wandb
        self.log(f"{stage}_macro_auroc", macro_auroc)
        self.log(f"{stage}_micro_auroc", micro_auroc)

        # except Exception as e:
        #     bp()
        #     print(f"Error calculating AUROC: {e}")

        return total_loss

    def log_gradient_stats(self):
        """
        Logs gradient statistics (mean, max, min) for each layer in the model.
        """
        gradient_stats = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                gradient_stats[f"grad_mean/{name}"] = param.grad.mean().item()
                gradient_stats[f"grad_max/{name}"] = param.grad.max().item()
                gradient_stats[f"grad_min/{name}"] = param.grad.min().item()

        # Log gradient stats to wandb
        wandb.log(gradient_stats)

    def on_after_backward(self):
        """Hook to log gradient stats after the backward pass."""

        if self.log_grads and self.current_epoch % 10 == 0:
            self.log_gradient_stats()

    def log_images_to_wandb(self, x_binary, x_hat_binary, x_real, x_hat_real, mask_binary, mask_real, stage):
        """
        Create imshow-style plots of the true x_binary and x_real vs reconstructed x_hat_binary and x_hat_real,
        with the mask applied as black color. Log them as a single figure to wandb.

        Args:
            x_binary (torch.Tensor): The true binary data.
            x_hat_binary (torch.Tensor): The reconstructed binary data.
            x_real (torch.Tensor): The true real-valued data.
            x_hat_real (torch.Tensor): The reconstructed real-valued data.
            mask_binary (torch.Tensor): The mask applied to binary data.
            mask_real (torch.Tensor): The mask applied to real-valued data.
            stage (str): Either 'train' or 'val', indicating the current stage of training.
        """

        if not self.log_plots:
            return

        # Convert to NumPy for plotting
        x_binary = x_binary.cpu().detach().numpy()
        x_hat_binary = x_hat_binary.cpu().detach().numpy()
        # convert to binary by thresholding at 0.5
        x_hat_binary_thresholded = (x_hat_binary > 0.5).astype(float)

        x_real = x_real.cpu().detach().numpy()
        x_hat_real = x_hat_real.cpu().detach().numpy()
        mask_binary = mask_binary.cpu().detach().numpy()
        mask_real = mask_real.cpu().detach().numpy()

        # Create a figure with 4 subplots: true and reconstructed for binary and real
        fig, ax = plt.subplots(3, 2, figsize=(18, 12))

        # Blue-White-Red colormap with black for masked regions
        cmap = plt.cm.bwr
        cmap.set_bad(color='black')

        # Define a function to replace masked areas with black
        def apply_mask(data, mask, vmin, vmax):
            masked_data = np.where(mask == 0, np.nan, data)  # Replace masked values with NaN
            return masked_data

        # Plot true binary data with mask applied
        true_binary_with_mask = apply_mask(x_binary, mask_binary, vmin=0, vmax=1)
        my_ax = ax[0, 0]
        my_ax.imshow(true_binary_with_mask, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        my_ax.set_title(f"True Binary Data ({stage})")
        my_ax.set_xlabel("Features")
        my_ax.set_ylabel("Samples")
        plt.colorbar(my_ax.imshow(true_binary_with_mask, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest'), ax=my_ax)

        # Plot reconstructed binary data with mask applied
        my_ax = ax[0, 1]
        reconstructed_logit_with_mask = apply_mask(x_hat_binary, mask_binary, vmin=0, vmax=1)
        my_ax.imshow(reconstructed_logit_with_mask, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        my_ax.set_title(f"Reconstructed Logit Data ({stage})")
        my_ax.set_xlabel("Features")
        plt.colorbar(my_ax.imshow(reconstructed_logit_with_mask, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest'), ax=my_ax)

        # Plot true binary data with mask applied
        my_ax = ax[1, 0]
        true_binary_with_mask = apply_mask(x_binary, mask_binary, vmin=0, vmax=1)
        my_ax.imshow(true_binary_with_mask, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        my_ax.set_title(f"True Binary Data ({stage})")
        my_ax.set_xlabel("Features")
        my_ax.set_ylabel("Samples")
        plt.colorbar(my_ax.imshow(true_binary_with_mask, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest'), ax=my_ax)

        # Plot reconstructed binary data with mask applied
        my_ax = ax[1, 1]
        reconstructed_binary_with_mask = apply_mask(x_hat_binary_thresholded, mask_binary, vmin=0, vmax=1)
        my_ax.imshow(reconstructed_binary_with_mask, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        my_ax.set_title(f"Reconstructed Binary Data ({stage})")
        my_ax.set_xlabel("Features")
        plt.colorbar(my_ax.imshow(reconstructed_binary_with_mask, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest'), ax=my_ax)

        # Plot true real data with mask applied
        my_ax = ax[2, 0]
        true_real_with_mask = apply_mask(x_real, mask_real, vmin=0, vmax=1)
        my_ax.imshow(true_real_with_mask, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        my_ax.set_title(f"True Real Data ({stage})")
        my_ax.set_xlabel("Features")
        my_ax.set_ylabel("Samples")
        plt.colorbar(my_ax.imshow(true_real_with_mask, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest'), ax=my_ax)

        # Plot reconstructed real data with mask applied
        my_ax = ax[2, 1]
        reconstructed_real_with_mask = apply_mask(x_hat_real, mask_real, vmin=0, vmax=1)
        my_ax.imshow(reconstructed_real_with_mask, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        my_ax.set_title(f"Reconstructed Real Data ({stage})")
        my_ax.set_xlabel("Features")
        plt.colorbar(my_ax.imshow(reconstructed_real_with_mask, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest'), ax=my_ax)

        # Adjust layout and log the figure to wandb
        plt.tight_layout()
        wandb.log({f"{stage}_reconstruction": wandb.Image(fig)})

        # Close the figure to free up memory
        plt.close(fig)


# 4. Training setup function with wandb logging
def train_model(model, train_loader, val_loader, test_loader, config):

    # Detect device: MPS for macOS, GPU for other systems if available
    if torch.backends.mps.is_available():
        config["accelerator"] = "mps"
    elif torch.cuda.is_available():
        config["accelerator"] = "gpu"
    else:
        config["accelerator"] = "cpu"

    output_dir = "outputs"
    os.makedirs(f"{output_dir}", exist_ok=True)

    wandb_logger = WandbLogger(project="ae_project", save_dir=output_dir)

    # Log the hyperparameters to wandb
    wandb_logger.experiment.config.update(config)

    trainer = pl.Trainer(
        logger=wandb_logger,
        log_every_n_steps=20, #Can modify this
        max_epochs=model.max_epochs,
        gradient_clip_val=config["gradient_clip_val"],
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval="step"),  # was "epoch"
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=config["early_stopping_patience"]),
            pl.callbacks.ModelCheckpoint(
                dirpath=f"{output_dir}/checkpoints/{wandb.run.name}",  # Directory to save checkpoints
                filename="{epoch}-{val_loss:.2f}",  # Naming pattern for checkpoint files
                save_top_k=1,  # Save top 3 checkpoints
                monitor="val_loss",  # Monitor validation loss
                mode="min",  # Save when validation loss decreases
            ),
        ],
        accelerator=config["accelerator"],  # mps, gpu, cpu
        # can configure devices= if desired to choose specific GPUs
    )
    trainer.fit(model, train_loader, val_loader)

    # Test the model on the test set
    trainer.test(model, test_loader)


#Uncomment this if you do not want to run as WandB sweep
# 5. Main function to run the script
if __name__ == "__main__":

    #Can later modify to use the argparser if needed

    # preliminary setup
    batch_size = 2048 # setting to None uses full dataset
    shuffle_train = True
    max_epochs = 1
    latent_dim = 500
    save_predictions = True
    test_only = False
    identity_run = False
    subset_data_30 = True
    subset_data_1_percent = False

    # Check if running in Google Colab
    in_colab = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ

    if in_colab:
        base_dir = "/content/drive/MyDrive/Research_Levine/diffusion_notebooks"
    else:
        base_dir = "data" #running locally


    #setup input and output directories and filenames
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

    print("batch_size:", batch_size)
    print("shuffle_train:", shuffle_train)
    print("max_epochs:", max_epochs)
    print("latent_dim:", latent_dim)
    print("save_predictions:", save_predictions)
    print("test_only:", test_only)
    print("identity_run:", identity_run)
    print("subset_data_30:", subset_data_30)
    print("subset_data_1_percent:", subset_data_1_percent)
    print("Input file path:", input_file)
    print("Output file path:", output_file)

    train_loader, val_loader, test_loader, scaler, col_list_dict, test_eir_suid, full_loader, all_eir_suid = prepare_data(
        input_file,
        batch_size,
        shuffle_train,
        test_dataset_predictions=test_only,
        test_dataset_raw_dir=test_dataset_raw_dir,
        input_type=output_file_postfix
    )

    # determine MSE vs BCE loss weights by counting number of features (and possibly doing further re-weighting)
    num_binary_features = train_loader.dataset.dataset.binary_data.shape[1]
    num_real_features = train_loader.dataset.dataset.real_data.shape[1]
    num_total_features = num_binary_features + num_real_features
    print(f"Input dimension: {num_total_features}")

    binary_weight = num_binary_features / num_total_features
    real_weight = 8.3 * num_real_features / num_total_features

    # then, further upweight the MSE loss to be of similar magnitude to BCE loss
    # e.g. mse_loss_weight = 8.3 allows losses to be equal when
    # the data are uniformly distributed between 0 and 1
    # and the BCE loss is 0.69 for a random prediction of 0.5
    # whereas the MSE loss for 0.5 would be 1/12 = 0.0833
    # their ratio is 0.69 / 0.0833 = 8.3

    # total_loss = real_weight * MSE_loss + binary_weight * BCE_loss

    # Define configuration
    # Base configuration with common values
    config = {
        "filename": input_file,
        "input_dim": num_total_features,
        "activation_fn": nn.PReLU,
        "use_residual": True,
        "use_batch_norm": True,
        "use_layer_norm": False,
        "learning_rate": 1e-4,
        "dropout_rate": 0.0,
        "batch_size": batch_size,  # None uses full dataset
        "shuffle_train": shuffle_train,
        "max_epochs": 1,
        "real_weight": real_weight,
        "binary_weight": binary_weight,
        "log_plot_freq": 20,
        "gradient_clip_val": 0,
        "log_plots": True,
        "log_grads": False,
        "early_stopping_patience": 10,
        "devices": [0]  # Use the first device
    }

    print(config)

    # Update specific values based on identity_run
    if identity_run:
        config.update({
            "latent_dim": num_total_features,
            "hidden_dims": [num_total_features],
            "identity_if_no_compression": True
        })
    else:
        config.update({
            "latent_dim": latent_dim,
            "hidden_dims": [12000, 8000, 4000, 2000, 1000],
            "identity_if_no_compression": False
        })


    # Initialize model
    model = ResidualAutoencoder(**config)

    # Train model
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


    '''
    NOTE:
    With latent_dim==input_dim and PReLU/GELU activation, we are able to get VERY close to identity mapping.
    A few comments:
        1. We are using residual connections (Projection based) and batch norm; can try without these.
        2. Needed to run for MANY epochs (set to 1000).

    '''
