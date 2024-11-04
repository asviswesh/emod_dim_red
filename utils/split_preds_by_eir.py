import os
import pandas as pd
import sys


latent_dim = 300
identity_run = False

# Check if running in Google Colab
in_colab = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ

# Provide the path to your CSV file, whether on Colab or local

if in_colab:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    base_dir = "/content/drive/MyDrive/Research_Levine/diffusion_notebooks/Predictions_Shared"
else:
    base_dir = "data/Predictions_Shared"


# Provide the path to your CSV file
if identity_run:
    input_file = f'{base_dir}/identity_run/predicted_full_dataset_identity_case_{latent_dim}_latent_dim_input_all_rows.csv'
    output_dir = f'{base_dir}/identity_run/split_by_eirs'
else:
    input_file = f'{base_dir}/latent_dim_{latent_dim}/predicted_full_dataset_{latent_dim}_latent_dim_input_all_rows.csv'
    output_dir = f'{base_dir}/latent_dim_{latent_dim}/split_by_eirs'


print(f"Input file: {input_file}")
print(f"Output directory: {output_dir}")
print(f"Latent dimension: {latent_dim}")
print(f"Identity run: {identity_run}")

def split_csv_by_eir(input_file, latent_dim, output_dir):
    # Load the CSV file
    df = pd.read_csv(input_file)
    original_row_count = len(df)

    # Extract filename and construct output directory based on latent dimension
    filename = os.path.basename(input_file)
    filename_without_ext = os.path.splitext(filename)[0]

    # Initialize a counter for total rows saved in split files
    total_rows_in_split_files = 0

    # Group by 'eir' column and save each group to a separate CSV file
    for eir_value, group in df.groupby('eir'):
        output_file = f'{output_dir}/predicted_full_dataset_{eir_value}_eir_{latent_dim}_latent_dim_input_all_rows.csv'
        group.to_csv(output_file, index=False)
        total_rows_in_split_files += len(group)  # Accumulate row count
        print(f'Saved {output_file}')

    # Verify row counts
    if total_rows_in_split_files == original_row_count:
        print(f"Row count verification successful: {original_row_count} rows in original file, {total_rows_in_split_files} rows in split files.")
    else:
        print(f"Row count mismatch: {original_row_count} rows in original file, {total_rows_in_split_files} rows in split files.")


split_csv_by_eir(input_file, latent_dim, output_dir)
