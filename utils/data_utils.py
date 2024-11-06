
# Utilities - data_utils.py

# import the necessary libraries
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split

from collections import defaultdict
from pdb import set_trace as bp

# Uncomment out below lines if using Colab
# # Check if CUDA is available
# if torch.cuda.is_available():
#     # Get the name of the current GPU
#     gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())

#     # Print the GPU name
#     print(f"Using GPU: {gpu_name}")

#     # Check if the GPU is one that supports Tensor Cores
#     if 'L4' in gpu_name or 'V100' in gpu_name or 'A100' in gpu_name:
#         # Set precision to high or medium for Tensor Cores optimization
#         precision_optmization='high'
#         torch.set_float32_matmul_precision(precision_optmization)
#         print(f"Setting torch.set_float32_matmul_precision to {precision_optmization} for Tensor Core optimization.")
#     else:
#         print("No Tensor Cores detected, no precision change applied.")
# else:
#     print("No CUDA device found, using CPU.")

# Flushing very small floating-point numbers is a recommended practice to improve the performance
# of neural network models. Flushing denormal numbers to zero, can prevent CPU computational slowdowns
# on certain hardware, especially for small-scale values in gradient updates or during optimization.
torch.set_flush_denormal(True)

class CustomDataset(Dataset):
    def __init__(self, data, binary_cols):

        if isinstance(data, pd.DataFrame):
            all_cols = data.columns
            real_cols = [col for col in all_cols if col not in binary_cols]
            binary_data = data[binary_cols].values.astype(float)
            real_data = data[real_cols].values.astype(float)
        else:
            all_cols = np.arange(data.shape[1])
            real_cols = list(set(all_cols) - set(binary_cols))
            binary_data = data[:, binary_cols].astype(float)
            real_data = data[:, real_cols].astype(float)

        self.binary_cols = binary_cols
        self.real_cols = real_cols

        # Create the mask before filling NaNs
        self.binary_mask = (~np.isnan(binary_data)).astype(float)  # Mask of available data (1 where data is present, 0 where it's NaN)
        self.real_mask = (~np.isnan(real_data)).astype(float)  # Mask of available data (1 where data is present, 0 where it's NaN)

        self.binary_data = np.nan_to_num(binary_data, nan=0.5)  # Replace NaNs with 0.5, after the mask is created
        self.real_data = np.nan_to_num(real_data, nan=0.5)  # Replace NaNs with 0.5, after the mask is created

        # print the overall shape of the data
        print("BINARY data report...")
        print(f"Data shape: {self.binary_data.shape}")

        # print the fraction of columns that have at least one NaN value
        print(f"Fraction of columns with NaNs: {np.mean(np.isnan(binary_data).any(axis=0))}")

        # Print the number of columns with NaNs
        print(f"Number of columns with NaNs: {np.isnan(binary_data).any(axis=0).sum()}")

        # print out each binary column that has missing values
        print("Columns with missing values:")
        for i, col in enumerate(binary_cols):
            if np.isnan(binary_data[:, i]).any():
                print(f"  - {col}")

        # print the fraction of rows that have at least one NaN value
        print(f"Fraction of rows with NaNs: {np.mean(np.isnan(binary_data).any(axis=1))}")

        # print the fraction of missing values in the dataset
        print(f"Overall fraction of missing values: {np.mean(np.isnan(binary_data))}")

        # print the overall shape of the data
        print("REAL data report...")
        print(f"Data shape: {self.real_data.shape}")

        # print the fraction of columns that have at least one NaN value
        print(f"Fraction of columns with NaNs: {np.mean(np.isnan(real_data).any(axis=0))}")

        # Print the number of columns with NaNs
        print(f"Number of columns with NaNs: {np.isnan(real_data).any(axis=0).sum()}")

        # print the fraction of rows that have at least one NaN value
        print(f"Fraction of rows with NaNs: {np.mean(np.isnan(real_data).any(axis=1))}")

        # print the fraction of missing values in the dataset
        print(f"Overall fraction of missing values: {np.mean(np.isnan(real_data))}")


    def __len__(self):
        return len(self.binary_data)

    def __getitem__(self, idx):
        # Return binary and real data, along with their respective masks
        x_binary = torch.tensor(self.binary_data[idx], dtype=torch.float32)
        x_real = torch.tensor(self.real_data[idx], dtype=torch.float32)
        binary_mask = torch.tensor(self.binary_mask[idx], dtype=torch.float32)
        real_mask = torch.tensor(self.real_mask[idx], dtype=torch.float32)

        return x_binary, x_real, binary_mask, real_mask


class CustomMinMaxScalerWithGroups:
    def __init__(self, feature_groups, feature_range=(0, 1)):
        """
        feature_groups: List of lists, where each sublist contains column names of features that should be treated as a group.
        Example: [['feature1', 'feature2'], ['feature3', 'feature4']]
        feature_range: Desired range of transformed data (default: (0, 1))
        """
        self.feature_groups = feature_groups
        self.feature_range = feature_range
        self.group_mins = {}
        self.group_maxs = {}
        self.columns_ = None  # To store column names when the input is a Pandas DataFrame
        self.feature_groups_indices = []  # To store column indices for each group

    # Fit the scaler by computing the min and max for each group of features (using column names)
    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns.tolist()

            # Convert feature groups from column names to indices
            self.feature_groups_indices = [
                [self.columns_.index(col) for col in group] for group in self.feature_groups
            ]

            X = X.values  # Convert to NumPy array for processing

        # Convert to float for log scaling
        X = X.astype(float)

        # Compute min and max for each group of features
        for i, group_indices in enumerate(self.feature_groups_indices):
            group_data = np.log10(1 + X[:, group_indices]) #apply a log10(1+x) transformation
            self.group_mins[i] = np.nanmin(group_data)
            self.group_maxs[i] = np.nanmax(group_data)
        return self

    # Transform the data by applying the min-max scaling to each group of features
    def transform(self, X, return_df=False):
        if not self.group_mins or not self.group_maxs:
            raise Exception("Scaler has not been fitted yet!")

        is_df = isinstance(X, pd.DataFrame)

        if is_df:
            X = X.values  # Convert DataFrame to NumPy array for transformation

        X_scaled = np.copy(X)
        min_r, max_r = self.feature_range  # Desired range of transformation

        # Convert to float for log scaling
        X = X.astype(float)

        # Assuming m_age is a known column name or index
        for i, group_indices in enumerate(self.feature_groups_indices):
            group_data = np.log10(1 + X[:, group_indices]) #apply a log10(1+x) transformation

            group_min = self.group_mins[i]
            group_max = self.group_maxs[i]

            # Standard min-max scaling to [0, 1]
            X_scaled[:, group_indices] = (group_data - group_min) / (group_max - group_min)

            # Rescale to desired feature_range (min_r, max_r)
            X_scaled[:, group_indices] = X_scaled[:, group_indices] * (max_r - min_r) + min_r

        if return_df and self.columns_:
            return pd.DataFrame(X_scaled, columns=self.columns_)

        return X_scaled

    # Fit and transform the data in one step
    def fit_transform(self, X, return_df=False):
        """Combines the fit and transform methods."""
        self.fit(X)
        return self.transform(X, return_df=return_df)

    # Reverse the scaling from transformed data back to the original scale
    def inverse_transform(self, X_scaled, return_df=False):
        if not self.group_mins or not self.group_maxs:
            raise Exception("Scaler has not been fitted yet!")

        is_df = isinstance(X_scaled, pd.DataFrame)

        if is_df:
            X_scaled = X_scaled.values  # Convert DataFrame to NumPy array for inverse transformation

        X_original = np.copy(X_scaled)
        min_r, max_r = self.feature_range  # Desired range of transformation

        for i, group_indices in enumerate(self.feature_groups_indices):

            group_data = X_scaled[:, group_indices]
            group_min = self.group_mins[i]
            group_max = self.group_maxs[i]

            # Rescale from [min_r, max_r] back to [0, 1]
            X_original[:, group_indices] = (group_data - min_r) / (max_r - min_r)

            # Reverse scaling from [0, 1] back to the original scale
            X_original[:, group_indices] = X_original[:, group_indices] * (group_max - group_min) + group_min

            # Undo the log10(1+x) transformation by applying 10^x - 1
            X_original[:, group_indices] = np.power(10, X_original[:, group_indices]) - 1

        if return_df and self.columns_:
            return pd.DataFrame(X_original, columns=self.columns_)

        return X_original


# 3. Data preprocessing and split
def prepare_data(filename="data/combined_data_subset_1_percent",
                 batch_size=None,
                 shuffle_train=True,
                 test_dataset_predictions=False,
                 test_dataset_raw_dir=None,
                 input_type="input_all_rows"):

    data = pd.read_csv(filename, low_memory=False)

    all_columns = data.columns

    # Keep track of "suid" and "eir" columns in order to add them back to the predictions
    eir_suid = data[['eir', 'suid']]

    # Drop columns "suid" and "eir" columns
    data = data.drop(columns=["suid", "eir"])

    # Keep track of all the columns and their ordering in order to return the predictions in the same order
    original_columns = data.columns

    # plot_missingness(data)

    # Identify columns that have only 1 unique value (i.e. degenerate columns) other than NaN, print them and drop
    single_value_cols = data.columns[data.nunique(dropna=True) == 1]

    # Keep track of all degenerate columns and their singular values in order to add them back to the predictions
    degenerate_cols = {col: data[col].dropna().unique()[0] if not data[col].dropna().empty else np.nan for col in single_value_cols}

    # Storing values of all degenerate columns
    degenerate_data = data[single_value_cols]

    # print all droppable columns even if it is a long list
    print("Droppable columns that are degenerate")
    for col in single_value_cols:
        print(f"  - {col} (constant value: {degenerate_cols[col]})")

    data = data.drop(columns=single_value_cols)
    print(f"Dropped {len(single_value_cols)} columns.")

    # Split into train/test
    # Generate a train-test split index, get the dataset based on the split indices, and ensure the split aligns with suid/eir
    train_idx, test_idx = train_test_split(data.index, test_size=0.2, random_state=42)

    # Use the split indices to create train and test datasets for both data and eir_suid
    train_data = data.loc[train_idx]
    test_data = data.loc[test_idx]

    # First get a list of all boolean columns by checking if all values are 0 or 1
    boolean_cols = [
        col
        for col in train_data.columns
        if set(train_data[col].dropna().unique()) == {0, 1}
    ]

    print(f"Found {len(boolean_cols)} non-degenerate boolean columns.")

    # Next, get a list of all columns that are bounded between 0 and 1 (but are not boolean)
    zero_to_one_cols = [
        col
        for col in train_data.columns
        if train_data[col].dropna().between(0, 1).all()
    ]
    print(f"Found {len(zero_to_one_cols)} non-degenerate zero_to_one columns.")

    # Now get all remaining columns
    remaining_cols = [
        col
        for col in train_data.columns
        if col not in boolean_cols and col not in zero_to_one_cols
    ]
    print(f"Found {len(remaining_cols)} non-degenerate remaining_cols > 1 value columns.")

    # Assert that none of the remaining columns have negative values
    assert all(
        train_data[remaining_cols].min() >= 0
    ), "Some columns have negative values!"

    # start with susceptibility columns that need grouping
    grouped_cols = {
        # MSP
        "m_MSP_antibodies_{i}_present": [
            f"m_MSP_antibodies_{i}_present" for i in range(32)
        ],
        "m_MSP_antibodies_{i}_m_antibody_capacity": [
            f"m_MSP_antibodies_{i}_m_antibody_capacity" for i in range(32)
        ],
        "m_MSP_antibodies_{i}_m_antibody_concentration": [
            f"m_MSP_antibodies_{i}_m_antibody_concentration" for i in range(32)
        ],
        # minor
        "m_PfEMP1_minor_antibodies_{i}_present": [
            f"m_PfEMP1_minor_antibodies_{i}_present" for i in range(385)
        ],
        "m_PfEMP1_minor_antibodies_{i}_m_antibody_capacity": [
            f"m_PfEMP1_minor_antibodies_{i}_m_antibody_capacity" for i in range(385)
        ],
        "m_PfEMP1_minor_antibodies_{i}_m_antibody_concentration": [
            f"m_PfEMP1_minor_antibodies_{i}_m_antibody_concentration"
            for i in range(385)
        ],
        # major
        "m_PfEMP1_major_antibodies_{i}_present": [
            f"m_PfEMP1_major_antibodies_{i}_present" for i in range(1070)
        ],
        "m_PfEMP1_major_antibodies_{i}_m_antibody_capacity": [
            f"m_PfEMP1_major_antibodies_{i}_m_antibody_capacity" for i in range(1070)
        ],
        "m_PfEMP1_major_antibodies_{i}_m_antibody_concentration": [
            f"m_PfEMP1_major_antibodies_{i}_m_antibody_concentration"
            for i in range(1070)
        ],
    }

    infection_fill_names = [
        "m_start_measuring",
        "m_is_symptomatic",
        "m_is_newly_symptomatic",
        "total_duration",
        "m_temp_duration",
        "m_measured_duration",
        "m_IRBCtimer",
        "m_gametosexratio",
        "m_gametorate",
        "infectiousness",
        "infectious_timer",
        "incubation_timer",
        "duration",
        "sim_time_created",
        "m_nonspectype",
        "m_MSPtype",
        "m_max_parasites",
        "m_inv_microliters_blood",
        "m_hepatocytes",
        "m_asexual_phase",
        "m_asexual_cycle_count",
    ]

    double_names = {
        "m_minor_epitope_type_{i}": [f"m_minor_epitope_type_{i}" for i in range(50)],
        "m_malegametocytes_{i}": [f"m_malegametocytes_{i}" for i in range(6)],
        "m_femalegametocytes_{i}": [f"m_femalegametocytes_{i}" for i in range(6)],
        "m_IRBCtype_{i}": [f"m_IRBCtype_{i}" for i in range(50)],
        "m_IRBC_count_{i}": [f"m_IRBC_count_{i}" for i in range(50)],
    }

    infection_list = [1, 2, 3]
    for nm in infection_fill_names:
        grouped_cols[nm] = [f"infection_{i}_{nm}" for i in infection_list]

    for nm in double_names:
        grouped_cols[nm] = [
            f"infection_{i}_{sub_nm}"
            for i in infection_list
            for sub_nm in double_names[nm]
        ]

    # Now, check whether any of the grouped columns are actually in single_value_cols
    # That is, the grouped columns is a hand-crafted list of columns.
    # The dataset may dictate that we throw out some of these columns.
    for group_name, group_cols in grouped_cols.items():
        if any(col in single_value_cols for col in group_cols):
            print(f"Group {group_name} contains single-value columns:")
            print([col for col in group_cols if col in single_value_cols])
            print("Removing them...")
            grouped_cols[group_name] = [
                col for col in group_cols if col not in single_value_cols
            ]

    # Check if any values in grouped_cols are empty lists. If so, remove them.
    empty_groups = [
        group_name for group_name, group_cols in grouped_cols.items() if not group_cols
    ]
    for group_name in empty_groups:
        print(f"Removing empty group: {group_name}")
        del grouped_cols[group_name]

    # build a single list of all columns that are grouped (i.e. unroll the dictionary)
    grouped_cols_list = [col for cols in grouped_cols.values() for col in cols]

    filtered_cols = data.columns # Columns of the original dataset before scaling.

    # now loop over all columns in the actual dataset.
    # if they are not in the grouped columns, then add them as their own group
    #i = 1 #Can this be removed
    for col in data.columns:
        if col not in grouped_cols_list:
            if col not in grouped_cols.keys():
                grouped_cols[col] = [col]
            else:
                # Key collision detected. Key is also a column, so append it to value
                print(f"Adding to {col} already present as a key in grouped_cols with values: {grouped_cols[col]}")
                grouped_cols[col].append(col)
            #i += 1 #Can this be removed

    print(f"We have distilled the dataset into {len(grouped_cols)} groups of columns.")
    print(
        "We will apply the same transformation to all columns in each group. This will help get robust statistics, and simplify our interpretations."
    )

    # We will now apply a MinMaxScaling that computes min/max values for each column group.
    # This will ensure that the same scaling is applied to all columns in a group.
    scaler = CustomMinMaxScalerWithGroups(
        feature_groups=list(grouped_cols.values()), feature_range=(0, 1)
    )
    train_data_scaled = scaler.fit_transform(train_data, return_df=True)
    test_data_scaled = scaler.transform(test_data, return_df=True)

    # Create Datasets without indices, working with NumPy arrays
    # re-build the grouped_cols_list
    all_cols = [col for cols in grouped_cols.values() for col in cols]

    columns_by_type = {
        "binary": [col for col in all_cols if col in boolean_cols],
        "real": [col for col in all_cols if col not in boolean_cols],
    }

    print("Creating training dataset...")
    train_dataset_pre_split = CustomDataset(
        train_data_scaled, binary_cols=columns_by_type["binary"]
    )

    print("Creating test dataset...")
    test_dataset = CustomDataset(
        test_data_scaled, binary_cols=columns_by_type["binary"]
    )


    # Original
    # # Split train data into train and validation: # 80% train, 20% validation
    # train_dataset, val_dataset = random_split(train_dataset_pre_split, [0.8, 0.2])

    # Split train data into train and validation: # 80% train, 20% validation
    # Set a fixed random seed for reproducibility
    seed = 42  # Choose any integer seed value

    # Create a torch generator with the seed
    generator = torch.Generator().manual_seed(seed)

    # Use the generator in random_split
    train_dataset, val_dataset = random_split(train_dataset_pre_split, [0.8, 0.2], generator=generator)

    if batch_size is None:
        train_batch_size = len(train_dataset)
        val_batch_size = len(val_dataset)
        test_batch_size = len(test_dataset)
    else:
        # take minimum of batch size and data size
        train_batch_size = min(len(train_dataset), batch_size)
        val_batch_size = min(len(val_dataset), batch_size)
        test_batch_size = min(len(test_dataset), batch_size)

    print(f"Using batch size of {train_batch_size} for training, {val_batch_size} for validation, and {test_batch_size} for testing.")

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=shuffle_train, num_workers=7)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=7)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=7)

    test_eir_suid = pd.DataFrame()
    full_loader = None
    if test_dataset_predictions:
        # Only need predictions of test data
        # We need to save the raw_test_data csv file to perform analysis on the test predictions

        # Add the eir and suid columns and degenerate columns to the test data
        test_eir_suid = eir_suid.loc[test_idx]
        test_data_with_eir_suid = pd.concat([test_data, test_eir_suid], axis=1)
        test_data_with_degenerate_cols = pd.concat([test_data_with_eir_suid, degenerate_data.loc[test_idx]], axis=1)
        test_data_with_degenerate_cols = test_data_with_degenerate_cols[all_columns]

        # Save the raw test data to csv file for perfoming analysis on test predictions later
        raw_test_data_filename =  f"{test_dataset_raw_dir}/raw_test_data_{input_type}.csv"
        print(f"Saving raw test data to {raw_test_data_filename}")
        test_data_with_degenerate_cols.to_csv(raw_test_data_filename, index=False)
    else:

        # Predictions of full dataset has to be returned. Currently needed by Rick Final
        full_data = data #Cannot copy data as we run out of memory
        full_data_scaled = scaler.transform(full_data, return_df=True)  # Transform entire dataset
        full_dataset = CustomDataset(full_data_scaled, binary_cols=columns_by_type["binary"])  # Create CustomDataset instance
        full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=7)  # Create DataLoader instance

    # This dictionary will help recreate the final predictions to have all columns as the in the raw data and in the same order
    col_list_dict = {"degenerate_cols": degenerate_cols,
                     "binary_cols": test_dataset.binary_cols,
                     "real_cols": test_dataset.real_cols,
                     "filtered_cols": filtered_cols,
                     "original_cols": original_columns,
                     "integer_encoded_binary_cols": ['m_gender']
                     }

     #Added full_loader for Rick final
    return train_loader, val_loader, test_loader, scaler, col_list_dict, test_eir_suid, full_loader, eir_suid


def find_problematic_columns(train_data, test_data, relative_threshold=2, absolute_threshold=1000):
    # Initialize empty lists to track columns with significant differences in min/max values
    problematic_columns = []

    # Iterate over each column
    for col in train_data.columns:
        # Get min/max for the training and test data for this column
        train_min, train_max = train_data[col].min(), train_data[col].max()
        test_min, test_max = test_data[col].min(), test_data[col].max()

        # Calculate relative differences
        relative_diff_max = (test_max - train_max) / (abs(train_max) + 1e-9)  # Adding a small number to avoid division by zero
        relative_diff_min = (train_min - test_min) / (abs(train_min) + 1e-9)

        # Check if the relative differences are above the threshold
        if abs(relative_diff_max) > relative_threshold or abs(relative_diff_min) > relative_threshold:
            problematic_columns.append({
                'column': col,
                'train_min': train_min,
                'train_max': train_max,
                'test_min': test_min,
                'test_max': test_max,
                'reason': 'Relative threshold exceeded'
            })

        # Check if the absolute differences exceed the absolute threshold
        elif abs(test_max - train_max) > absolute_threshold or abs(test_min - train_min) > absolute_threshold:
            problematic_columns.append({
                'column': col,
                'train_min': train_min,
                'train_max': train_max,
                'test_min': test_min,
                'test_max': test_max,
                'reason': 'Absolute threshold exceeded'
            })

    # Print out the problematic columns
    if problematic_columns:
        print("Problematic columns (where test values differ significantly from training values):")
        for col_info in problematic_columns:
            print(f"Column: {col_info['column']}")
            print(f"  Reason: {col_info['reason']}")
            print(f"  Train Min: {col_info['train_min']}, Train Max: {col_info['train_max']}")
            print(f"  Test Min: {col_info['test_min']}, Test Max: {col_info['test_max']}")
    else:
        print("No problematic columns found.")


def reverse_normalization(reconstructed_binary, reconstructed_real, scaler, col_list_dict):
    """
    Use the scaler's inverse_transform method to reverse the normalization.
    """
    # Combine the binary and real data and inverse transform them
    combined_data = np.concatenate([reconstructed_binary.cpu().numpy(), reconstructed_real.cpu().numpy()], axis=1)

    combined_data_cols = col_list_dict["binary_cols"] + col_list_dict["real_cols"]
    combined_data_df = pd.DataFrame(combined_data)
    combined_data_df.columns = combined_data_cols

    denormalized_data_array = combined_data_df.reindex(columns=col_list_dict["filtered_cols"]).to_numpy()

    # Create an empty array to store the denormalized data
    denormalized_data = np.zeros_like(denormalized_data_array)

    # Use the scaler's inverse_transform to bring back the original scale
    denormalized_data = scaler.inverse_transform(denormalized_data_array)

    return denormalized_data


def insert_degenerate_columns(denormalized_data, col_list_dict, scaler):
    """
    Insert degenerate columns back into the denormalized dataset and reorder it based on the original columns.
    """

    if hasattr(scaler, 'columns_'):
        column_names = scaler.columns_

        # Convert combined_predictions into a DataFrame with the original column names
        denormalized_df = pd.DataFrame(denormalized_data, columns=column_names)

    # Create a DataFrame from the degenerate columns
    degenerate_cols_df = pd.DataFrame(col_list_dict["degenerate_cols"], index=denormalized_df.index)

    # Concatenate the degenerate columns with the denormalized DataFrame
    denormalized_df = pd.concat([denormalized_df, degenerate_cols_df], axis=1)

    # Reorder the columns to match the original dataset's column order
    denormalized_df = denormalized_df.reindex(columns=col_list_dict["original_cols"])

    return denormalized_df

def create_and_populate_dependency_dict(masked_dict, all_columns):
    """
    Creates keys for the independent columns and populates masked_dict with
    dependent columns (antibody/infection present columns and their corresponding
    capacity/concentration columns).
    """

    # Handle MSP, PfEMP1_minor, and PfEMP1_major antibodies
    antibody_types = {
        'm_MSP_antibodies': 32,
        'm_PfEMP1_minor_antibodies': 385,
        'm_PfEMP1_major_antibodies': 1070
    }

    for antibody_type, count in antibody_types.items():
        for i in range(count):
            key = f'{antibody_type}_{i}_present' #add key
            masked_dict[key].extend([
                f'{antibody_type}_{i}_m_antibody_capacity',
                f'{antibody_type}_{i}_m_antibody_concentration'
            ])

    # Handle infections
    for col in all_columns:
        # There are 3 types of infections.
        for i in range(1, 4):
            prefix = f'infection_{i}_'
            if col.startswith(prefix) and 'present' not in col:
                masked_dict[f'infection_{i}_present'].append(col)
                break


def print_nan_stats(masked_predictions_df, col_list_dict):
    """
    Computes various statistics on the data after filtering out degenerate columns.
    It calculates the shape, total number of NaN values, fraction of NaN values,
    number of columns and rows with NaNs, and the percentage of columns and rows with NaNs.

    """

    filtered_predictions_df = masked_predictions_df.drop(columns=col_list_dict["degenerate_cols"])
    filtered_predictions = filtered_predictions_df.to_numpy()

    shape_of_filtered_predictions = filtered_predictions.shape
    num_nans = np.isnan(filtered_predictions.astype(float)).sum()
    num_columns_with_nan = np.isnan(filtered_predictions.astype(float)).any(axis=0).sum()
    percentage_columns_with_nan = np.isnan(filtered_predictions.astype(float)).any(axis=0).mean() * 100
    num_rows_with_nan = np.isnan(filtered_predictions.astype(float)).any(axis=1).sum()
    percentage_rows_with_nan = np.isnan(filtered_predictions.astype(float)).any(axis=1).mean() * 100
    fraction_nans = num_nans / filtered_predictions.size

    print(f"Statistics of Test Data:")
    print(f"Shape of the test data: [{shape_of_filtered_predictions}]")
    print(f"Total Number of NaN values: {num_nans}")
    print(f"Overall fraction of NaN values: {fraction_nans}")
    print(f"Total number of columns in the data with NaN: {num_columns_with_nan}")
    print(f"Fraction of columns in the data with NaN: {percentage_columns_with_nan}")
    print(f"Total number of rows in the data with NaN: {num_rows_with_nan}")
    print(f"Fraction of rows in the data with NaN: {percentage_rows_with_nan}")


def threshold_binary_columns(col_list_dict, reconstructed_df):
    """
    Apply thresholding to binary columns specified in col_list_dict["binary_cols"].
    Binary columns will be thresholded such that values >= 0.5 are set to True, and others to False.
    """
    binary_cols = col_list_dict.get("binary_cols", [])
    #filtered_cols = col_list_dict.get("filtered_cols", [])
    integer_encoded_binary_cols = col_list_dict.get("integer_encoded_binary_cols", [])

    if not binary_cols:
        print("No binary columns found in col_list_dict['binary_cols'].")
        return

    # Determine binary columns that are not in filtered_cols
    #binary_cols_to_threshold = [col for col in binary_cols if col not in filtered_cols]
    binary_cols_to_threshold = [col for col in binary_cols if col not in integer_encoded_binary_cols]

    # Apply thresholding (>= 0.5) to all binary columns (set values to 1 or 0)
    reconstructed_df[binary_cols] = (reconstructed_df[binary_cols] >= 0.5).astype(int)

    reconstructed_df[binary_cols_to_threshold] = reconstructed_df[binary_cols_to_threshold].astype(bool)


    return reconstructed_df


def convert_to_nan_based_on_dependencies(reconstructed_df):
    """
    Applies a dependency mask and NaN conversion to a DataFrame.
    This includes creating a dictionary to track dependent columns that can contain NaN values,
    and applying NaN conversion based on it.
    """

    # Create a dictionary to track dependent columns that may contain NaN values
    masked_dict = defaultdict(list)
    create_and_populate_dependency_dict(masked_dict, reconstructed_df.columns)

    # Iterate over the independent-dependent column mappings in masked_dict
    for ind_col, dep_cols in masked_dict.items():
        reconstructed_df.loc[reconstructed_df[ind_col] == 0, dep_cols] = np.nan

    return reconstructed_df


def add_eir_suid_to_predictions(reconstructed_df, test_eir_suid, col_list_dict):
    """
    Adds 'eir' and 'suid' columns back to the masked_predictions DataFrame
    and reconstructs the full DataFrame with appropriate column names.
    """

    reconstructed_cols_list = ['eir', 'suid'] + col_list_dict["original_cols"].tolist()
    post_processed_df = pd.concat([test_eir_suid.reset_index(drop=True), reconstructed_df], axis=1)
    post_processed_df.columns = reconstructed_cols_list #Assign the reconstructed column names
    return post_processed_df


def post_process_predictions(x_hat_binary, x_hat_real, scaler, col_list_dict, test_eir_suid):
    """
    Post-processes the model output by reversing normalization, rounding binary values,
    adding degenerate columns and reverting to original column ordering,
    applying a dependency mask for NaN conversion of some values, and adding the columns
    eir, suid back as the first 2 columns.

    Parameters:
    -----------
    x_hat_binary : numpy.ndarray
        The predicted binary values from the model output.

    x_hat_real : numpy.ndarray
        The predicted real values from the model output.

    scaler : object
        The scaler used for normalization, to reverse the normalization process.

    col_list_dict : dict
        A dictionary containing column information including original columns and degenerate columns.

    test_eir_suid : pd.DataFrame
        A DataFrame containing the 'eir' and 'suid' columns. These columns serve as
        identifiers for the data and will be added back to the predictions.

    Returns:
    --------
    numpy.ndarray
        The final post-processed predictions, including NaN conversions where appropriate.
    """

    # STEP 1: Reverse the normalization
    denorm_df = reverse_normalization(x_hat_binary, x_hat_real, scaler, col_list_dict)

    # STEP 2: Put back the degenerate columns
    reconstructed_df = insert_degenerate_columns(denorm_df, col_list_dict, scaler)

    # Step 3: Convert binary values to 0s and 1s
    reconstructed_df_with_bin_threshold = threshold_binary_columns(col_list_dict, reconstructed_df)

    # STEP 4: Apply NaN conversion
    reconstructed_df = convert_to_nan_based_on_dependencies(reconstructed_df_with_bin_threshold)

    # STEP 5: Add the eir and suid back to the data
    final_predictions_df = add_eir_suid_to_predictions(reconstructed_df, test_eir_suid, col_list_dict)

    return final_predictions_df
