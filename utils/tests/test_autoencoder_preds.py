#TESTS
import pandas as pd
import numpy as np
import json
import sys
import os

def add_eir_and_suid_to_beginning(original_columns):
    """
    Helper Function - Adds eir and suid columns to the beginning of original_columns.
    """
    # Convert original_columns to a list if it's a Pandas Index
    if isinstance(original_columns, pd.Index):
        original_columns_list = list(original_columns)
    else:
        original_columns_list = original_columns

    # Add new columns to the beginning
    updated_columns_list = ['eir', 'suid'] + original_columns_list

    # Convert back to a Pandas Index and return
    return pd.Index(updated_columns_list)


def obtain_bin_real_cols(data, col_list_dict):
    """
    Helper function to obtain binary and real-valued columns from a CSV file afte dropping eir, suid, and degenerate columns.
    """

    if "suid" in data.columns and "eir" in data.columns:
        data = data.drop(columns=["suid", "eir"])


    single_value_cols = data.columns[data.nunique(dropna=True) == 1]


    data = data.drop(columns=single_value_cols)

    binary_cols = [
        col
        for col in data.columns
        if set(data[col].dropna().unique()) == {0, 1}
    ]

    real_cols = [col for col in data.columns if col not in binary_cols]

    return binary_cols, real_cols


def compare_and_print_differences(orig_binary_cols_set, pred_binary_cols_set, col_type):
    """
    Helper function to compare two sets and print if there are any the differences.
    """
    # Elements in set1 but not in set2
    in_set1_not_in_set2 = orig_binary_cols_set - pred_binary_cols_set
    # Elements in set2 but not in set1
    in_set2_not_in_set1 =  pred_binary_cols_set - orig_binary_cols_set

    match_set1 = False
    if in_set1_not_in_set2:
        print(f"{col_type} columns in raw but not in preds: {in_set1_not_in_set2}")
    else:
        print(f"No {col_type} columns in raw that are missing from preds.")
        match_set1 = True

    match_set2 = False
    if in_set2_not_in_set1:
        print(f"{col_type} columns in preds but not in raw: {in_set2_not_in_set1}")
    else:
        print(f"No {col_type} columns in preds that are missing from raw.")
        match_set2 = True

    return match_set1 and match_set2


def print_column_stats(preds, real_cols, tolerances):
    for column, tolerance in zip(real_cols, tolerances):
        if column in preds.columns:
            min_value = preds[column].min()
            max_value = preds[column].max()
            median_value = preds[column].median()
            mean_value = preds[column].mean()
            print(f"Column: {column} - Min: {min_value}, Max: {max_value}, "
                  f"Median: {median_value}, Mean: {mean_value}, Tolerance: {tolerance}")



def compare_dataframes(orig, preds, binary_cols, real_cols, tolerance=0.0001):
    """
    Compares two DataFrames allowing a tolerance for real columns and exact match for binary columns.
    Prints the first mismatched value when a difference is found.
    """
    real_response=True
    # Compare real columns with tolerance
    for col in real_cols:
        if not np.allclose(orig[col], preds[col], atol=tolerance, rtol = 0, equal_nan=True):
            # Find the first mismatched value
            diff = np.abs(orig[col] - preds[col]) > tolerance
            first_mismatch_index = diff.idxmax()  # Get index of the first mismatch
            print(f"Difference in real column: {col}")
            orig_real_value = orig[col].iloc[first_mismatch_index]
            pred_real_value = preds[col].iloc[first_mismatch_index]
            diff_real = np.abs(orig_real_value - pred_real_value)
            print(f"Real Cols: First mismatch at index {first_mismatch_index}: Original value = {orig_real_value}, Prediction value = {pred_real_value}, diff = {diff_real}")
            real_response=False
            break

    if real_response:
      print(f"No differences found in real values between raw and predicted")

    # Compare binary columns with a tolerance for values close to 1 and 0
    # NOTE: binary columns can be T/F or 0 and 1
    binary_response = True
    for col in binary_cols:
        # Round values close to 1 and 0 to handle binary precision issues
        orig_binary = orig[col].fillna(np.nan).apply(lambda x: 1 if x >= (1 - tolerance) else 0 if x <= tolerance else np.nan).astype('Int64')
        preds_binary = preds[col].fillna(np.nan).apply(lambda x: 1 if x >= (1 - tolerance) else 0 if x <= tolerance else np.nan).astype('Int64')

        # Check for mismatches
        mismatches = orig_binary != preds_binary

        if mismatches.any():
            first_mismatch_index = mismatches.idxmax()  # Get index of the first mismatch
            print(f"Difference in binary column: {col}")
            orig_binary_value = orig_binary.iloc[first_mismatch_index]
            pred_binary_value = preds_binary.iloc[first_mismatch_index]
            diff = np.abs(orig_binary_value - pred_binary_value)
            print(f"First mismatch at index {first_mismatch_index}: Original value = {orig_binary_value}, Prediction value = {pred_binary_value}, diff = {diff}")
            response=False
            binary_response = False
            break

    if binary_response:
      print(f"No differences found in binary values between raw and predicted")

    return binary_response and real_response


def verify_count_of_all_columns(original_columns, pred_columns):
    """
    Verify that number of columns in raw dataset matches number of predicted columns.
    """
    # original_columns = add_eir_and_suid_to_beginning(original_columns)
    #pred_columns = pred_columns[2:] #compare with eir and suid
    return len(original_columns) == len(pred_columns)


def verify_column_order_of_all_columns(original_columns, pred_columns):
    """
    Verify that all the columns in the raw data match all the predicted columns in the exact same order
    includes eir, suid, degenerate columns i.e. all columns in the raw data set are included.
    """
    # original_columns = add_eir_and_suid_to_beginning(original_columns)
    #pred_columns = pred_columns[2:] #compare with eir and suid
    for i in range(len(original_columns)):
        if original_columns[i] != pred_columns[i]:
            return False

    return True

def verify_count_of_binary_and_real_columns(orig, preds, col_list_dict):
    """
    Verify that the original and predicted data contain the same binary and real-valued columns for identity case.
    """

    raw_bin_cols, raw_real_cols = obtain_bin_real_cols(orig, col_list_dict)
    pred_bin_cols, pred_real_cols = obtain_bin_real_cols(preds, col_list_dict)

    pred_binary_cols_set = set(pred_bin_cols)
    pred_real_cols_set = set(pred_real_cols)

    # Check if the column exists in the set
    orig_binary_cols_set = set(raw_bin_cols)
    # Maybe we should explicitly track all binary columns instead of treating columns as binary if their values are 0 and 1? then we will not have these problems
    orig_binary_cols_set.discard('infection_1_m_IRBC_count_43')
    raw_real_cols.append('infection_1_m_IRBC_count_43')
    orig_real_cols_set = set(raw_real_cols)

    # binary_cols_match = (pred_binary_cols_set == orig_binary_cols_set)
    # real_cols_match = (pred_real_cols_set == orig_real_cols_set)

    binary_cols_match = compare_and_print_differences(orig_binary_cols_set, pred_binary_cols_set, "binary")
    real_cols_match = compare_and_print_differences(orig_real_cols_set, pred_real_cols_set, "real")

    return binary_cols_match, real_cols_match




def verify_predicted_values_with_raw_for_identity_case(orig_dropped, preds, tolerance=0):
    """Verify that the raw data matches predicted data when the 'eir' and 'suid' columns are removed,
    and save the modified CSV."""

    pred_minus_eir_suid = preds.drop(columns=['eir', 'suid'], errors='ignore') #drop eir and suid
    #drop degenerate columns
    #Changed, since degenerate_cols is already a list with keys
    #degenerate_cols = list(col_list_dict["degenerate_cols"].keys())
    degenerate_cols = col_list_dict["degenerate_cols"]
    pred_dropped = pred_minus_eir_suid.drop(columns=degenerate_cols, errors='ignore')

    # Call the comparison function
    real_cols_list = col_list_dict["real_cols"]
    binary_cols_list = col_list_dict["binary_cols"]
    are_equal = compare_dataframes(orig_dropped, pred_dropped, binary_cols_list, real_cols_list, tolerance)
    return are_equal



def print_differences(original_df, predictions_df, columns, tolerance):
    for column in columns:
        if column in original_df.columns and column in predictions_df.columns:
            # Get original and predicted values
            original_values = original_df[column]
            predicted_values = predictions_df[column]

            # Calculate differences
            differences = (original_values - predicted_values).abs()

            # Find indices where the difference exceeds the tolerance
            significant_differences = differences[differences > tolerance]

            # Print results
            if not significant_differences.empty:
                print(f"\nColumn: {column}")
                for index in significant_differences.index:
                    orig_value = original_values[index]
                    pred_value = predicted_values[index]
                    diff_value = significant_differences[index]
                    print(f"Index: {index} - Original: {orig_value}, Predicted: {pred_value}, Difference: {diff_value}")

def normalize_binary_values(df):
    """
    Normalize binary columns to 0 and 1.
    Converts True/False to 1/0 and keeps 0/1 as is.
    """
    return df.applymap(lambda x: 1 if x is True else (0 if x is False else x))


def print_real_difference_summary(original_df, predictions_df, columns, tolerance, output_file, print_all_stats=False):
    count_with_differences = 0
    count_without_differences = 0
    total_differences = 0  # Initialize total differences count
    total_values = 0  # Initialize total values count

    # Initialize variables for lowest and highest differences
    lowest_difference = float('inf')
    highest_difference = float('-inf')
    lowest_info = {}
    highest_info = {}

    summary_stats = []  # List to store summary stats for each column

    for column in columns:
        if column in original_df.columns and column in predictions_df.columns:
            # Get original and predicted values
            original_values = original_df[column]
            predicted_values = predictions_df[column]

            # Calculate absolute differences
            differences = (original_values - predicted_values).abs()

            # Find significant differences
            significant_differences = differences[differences > tolerance]

            # Update total values count
            total_values += original_values.size

            # Calculate mean, median, min, and max for original and predicted values
            mean_original = original_values.mean()
            median_original = original_values.median()
            min_original = original_values.min()
            max_original = original_values.max()

            mean_predicted = predicted_values.mean()
            median_predicted = predicted_values.median()
            min_predicted = predicted_values.min()
            max_predicted = predicted_values.max()

            # Calculate min and max differences and count of significant differences
            if not significant_differences.empty:
                min_diff = significant_differences.min()
                max_diff = significant_differences.max()
                count_diff = significant_differences.count()
                total_differences += count_diff  # Update total differences count

                # Get the indices of min and max differences
                min_diff_index = significant_differences.idxmin()
                max_diff_index = significant_differences.idxmax()

                # Calculate percentage errors
                min_percentage_error = (min_diff / original_values[min_diff_index]) * 100 if original_values[min_diff_index] != 0 else 0
                max_percentage_error = (max_diff / original_values[max_diff_index]) * 100 if original_values[max_diff_index] != 0 else 0

                # Calculate error rate for the column
                error_rate = (count_diff / total_values) * 100

                # Calculate NaN percentages
                percent_nans_original = original_values.isna().mean() * 100
                percent_nans_predicted = predicted_values.isna().mean() * 100

                # Store info for the lowest and highest differences
                if min_diff < lowest_difference:
                    lowest_difference = min_diff
                    lowest_info = {
                        'raw_value': original_values[min_diff_index],
                        'predicted_value': predicted_values[min_diff_index],
                        'difference': min_diff,
                        'percentage_error': min_percentage_error,
                        'column': column
                    }

                if max_diff > highest_difference:
                    highest_difference = max_diff
                    highest_info = {
                        'raw_value': original_values[max_diff_index],
                        'predicted_value': predicted_values[max_diff_index],
                        'difference': max_diff,
                        'percentage_error': max_percentage_error,
                        'column': column
                    }

                # Append summary stats to list
                summary_stats.append({
                    'Column': column,
                    'Min Difference': min_diff,
                    'Max Difference': max_diff,
                    'Mean Original': mean_original,
                    'Median Original': median_original,
                    'Min Original': min_original,
                    'Max Original': max_original,
                    'Mean Predicted': mean_predicted,
                    'Median Predicted': median_predicted,
                    'Min Predicted': min_predicted,
                    'Max Predicted': max_predicted,
                    'Count of Differences': count_diff,
                    'Max Percentage Error': f"{max_percentage_error:.8f}",
                    'Error Rate': error_rate,
                    'Percent NaNs Original': percent_nans_original,
                    'Percent NaNs Predicted': percent_nans_predicted
                })

                if print_all_stats:
                    # Print the summary for the column
                    print(
                        f"Column: {column} - Min: {min_original}, Max: {max_original}, "
                        f"Mean Original: {mean_original}, Median Original: {median_original}, "
                        f"Mean Predicted: {mean_predicted}, Median Predicted: {median_predicted}, "
                        f"Min Difference: {min_diff}, Max Difference: {max_diff}, "
                        f"Count of Differences: {count_diff}, Max Percentage Error: {max_percentage_error:.8f}%, "
                        f"Error Rate: {error_rate:.2f}%, "
                        f"% NaNs Original: {percent_nans_original:.2f}%, % NaNs Predicted: {percent_nans_predicted:.2f}%"
                    )

                count_with_differences += 1
            else:
                count_without_differences += 1

    # Calculate percentages
    percentage_with_differences = (total_differences / total_values) * 100 if total_values > 0 else 0
    percentage_without_differences = ((total_values - total_differences) / total_values) * 100 if total_values > 0 else 0

    # Overall summary stats
    overall_summary_stats = {
        'Total Columns with Differences': count_with_differences,
        'Total Columns without Differences': count_without_differences,
        'Total Values with Differences': total_differences,
        'Total Values without Differences': total_values - total_differences,
        '% Values with Differences': percentage_with_differences,
        '% Values without Differences': percentage_without_differences
    }

    # Print overall summary stats
    print("\nOverall Summary Stats:")
    for key, value in overall_summary_stats.items():
        print(f"{key}: {value}")

    # Add overall summary to the beginning of the summary stats
    summary_stats.insert(0, overall_summary_stats)

    # Write summary stats to a CSV file
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_file, index=False)
    print(f"\nSummary statistics saved to {output_file}")

def print_binary_difference_summary(original_df, predictions_df, columns, tolerance, output_file, print_all_stats=False):
    count_with_differences = 0
    count_without_differences = 0
    total_differences = 0  # Initialize total differences count
    total_values = 0  # Initialize total values count

    # Initialize variables for lowest and highest differences
    lowest_difference = float('inf')
    highest_difference = float('-inf')
    lowest_info = {}
    highest_info = {}

    # Initialize a list to store summary stats for each column
    summary_stats = []

    # Normalize binary values in both dataframes if data_type is binary
    original_df = normalize_binary_values(original_df[columns])
    predictions_df = normalize_binary_values(predictions_df[columns])

    for column in columns:
        if column in original_df.columns and column in predictions_df.columns:
            # Get original and predicted values
            original_values = original_df[column]
            predicted_values = predictions_df[column]

            # Calculate absolute differences
            differences = (original_values - predicted_values).abs()

            # Calculate error percentage for the column
            error_count = differences.gt(tolerance).sum()
            total_column_values = len(differences)
            error_percentage = (error_count / total_column_values) * 100

            # Calculate percentages for binary data (0, 1, NaN) in both raw and predicted
            percent_zeros_raw = (original_values == 0).mean() * 100
            percent_ones_raw = (original_values == 1).mean() * 100
            percent_nans_raw = 100 - (percent_zeros_raw + percent_ones_raw)

            percent_zeros_pred = (predicted_values == 0).mean() * 100
            percent_ones_pred = (predicted_values == 1).mean() * 100
            percent_nans_pred = 100 - (percent_zeros_pred + percent_ones_pred)

            # Collect basic stats for raw and predicted values
            stats = {
                "Column": column,
                "Min Raw": original_values.min(),
                "Max Raw": original_values.max(),
                "Mean Raw": original_values.mean(),
                "Median Raw": original_values.median(),
                "Min Predicted": predicted_values.min(),
                "Max Predicted": predicted_values.max(),
                "Mean Predicted": predicted_values.mean(),
                "Median Predicted": predicted_values.median(),
                "Percent 0s Raw": percent_zeros_raw,
                "Percent 1s Raw": percent_ones_raw,
                "Percent NaNs Raw": percent_nans_raw,
                "Percent 0s Predicted": percent_zeros_pred,
                "Percent 1s Predicted": percent_ones_pred,
                "Percent NaNs Predicted": percent_nans_pred,
                "Error Percentage": error_percentage  # Add error percentage to stats
            }

            if print_all_stats:
                # Print all stats in a single line for the column
                print(
                    f"Column: {column} | Min Raw: {stats['Min Raw']} | Max Raw: {stats['Max Raw']} | "
                    f"Mean Raw: {stats['Mean Raw']:.2f} | Median Raw: {stats['Median Raw']:.2f} | "
                    f"Min Predicted: {stats['Min Predicted']} | Max Predicted: {stats['Max Predicted']} | "
                    f"Mean Predicted: {stats['Mean Predicted']:.2f} | Median Predicted: {stats['Median Predicted']:.2f} | "
                    f"%0s Raw: {stats['Percent 0s Raw']:.2f}% | %1s Raw: {stats['Percent 1s Raw']:.2f}% | %NaNs Raw: {stats['Percent NaNs Raw']:.2f}% | "
                    f"%0s Predicted: {stats['Percent 0s Predicted']:.2f}% | %1s Predicted: {stats['Percent 1s Predicted']:.2f}% | %NaNs Predicted: {stats['Percent NaNs Predicted']:.2f}% | "
                    f"Error Percentage: {stats['Error Percentage']:.2f}%"
                )

            # Append stats to the summary
            summary_stats.append(stats)

            # Track differences
            if error_count > 0:
                count_with_differences += 1
                total_differences += error_count
            else:
                count_without_differences += 1

            total_values += total_column_values

    # Calculate percentages for total differences
    percentage_with_differences = (total_differences / total_values) * 100 if total_values > 0 else 0
    percentage_without_differences = ((total_values - total_differences) / total_values) * 100 if total_values > 0 else 0

    # Overall summary stats
    overall_summary_stats = {
        'Total Columns with Differences': count_with_differences,
        'Total Columns without Differences': count_without_differences,
        'Total Values with Differences': total_differences,
        'Total Values without Differences': total_values - total_differences,
        '% Values with Differences': percentage_with_differences,
        '% Values without Differences': percentage_without_differences
    }

    # Print overall summary stats
    print("\nOverall Summary Stats:")
    for key, value in overall_summary_stats.items():
        print(f"{key}: {value}")

    # Add overall summary to the beginning of the summary stats
    summary_stats.insert(0, overall_summary_stats)

    # Write summary stats to a CSV file
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_file, index=False)
    print(f"\nSummary statistics saved to {output_file}")

# Test

if __name__ == "__main__":

    # Check if running in Google Colab
    in_colab = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ

    if in_colab:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
        base_dir = "/content/drive/MyDrive/Research_Levine/diffusion_notebooks"
    else:
        base_dir = "../../data"

    latent_dim = 300
    test_only = False
    identity_run = False
    subset_data_30 = True
    subset_data_1_percent = False

    #Set this to true if you want to also print on the output each column stats that will be written to the "difference" files
    print_all_stats = False


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

    #setup input and output directories
    if test_only:
        test_dataset_raw_dir="/content/drive/MyDrive/Research_Levine/diffusion_notebooks/Inputs/generated_test_data"
        input_file =  f"{test_dataset_raw_dir}/raw_test_data_{output_file_postfix}.csv"
        output_test_dataset_dir = f"{output_dir}/test_dataset"
        output_test_dataset_dir_performance_analysis = f"{output_test_dataset_dir}/performance_analysis"
        if identity_run:
            output_file = f"{output_test_dataset_dir}/identity/predicted_test_dataset_identity_case_{latent_dim}_latent_dim_{output_file_postfix}.csv"
            performance_analysis_binary_file = f"{output_test_dataset_dir_performance_analysis}/binary_cols_performance_analysis_test_dataset_identity_case_{latent_dim}_latent_dim_{output_file_postfix}.csv"
            performance_analysis_real_file = f"{output_test_dataset_dir_performance_analysis}/real_cols_performance_analysis_test_dataset_identity_case_{latent_dim}_latent_dim_{output_file_postfix}.csv"
        else:
            output_file = f"{output_test_dataset_dir}/{latent_dim}/predicted_test_dataset_{latent_dim}_latent_dim_{output_file_postfix}.csv"
            performance_analysis_binary_file = f"{output_test_dataset_dir_performance_analysis}/binary_cols_performance_analysis_test_dataset_{latent_dim}_latent_dim_{output_file_postfix}.csv"
            performance_analysis_real_file = f"{output_test_dataset_dir_performance_analysis}/real_cols_performance_analysis_test_dataset_{latent_dim}_latent_dim_{output_file_postfix}.csv"
    else:

        output_full_dataset_dir = f"{output_dir}/full_dataset"
        output_full_dataset_dir_performance_analysis = f"{output_full_dataset_dir}/performance_analysis"
        if identity_run:
            output_file = f"{output_full_dataset_dir}/identity/predicted_full_dataset_identity_case_{latent_dim}_latent_dim_{output_file_postfix}.csv"
            performance_analysis_binary_file = f"{output_full_dataset_dir_performance_analysis}/binary_cols_performance_analysis_full_dataset_identity_case_{latent_dim}_latent_dim_{output_file_postfix}.csv"
            performance_analysis_real_file = f"{output_full_dataset_dir_performance_analysis}/real_cols_performance_analysis_full_dataset_identity_case_{latent_dim}_latent_dim_{output_file_postfix}.csv"
        else:
            output_file = f"{output_full_dataset_dir}/{latent_dim}/predicted_full_dataset_{latent_dim}_latent_dim_{output_file_postfix}.csv"
            performance_analysis_binary_file = f"{output_full_dataset_dir_performance_analysis}/binary_cols_performance_analysis_full_dataset_{latent_dim}_latent_dim_{output_file_postfix}.csv"
            performance_analysis_real_file = f"{output_full_dataset_dir_performance_analysis}/real_cols_performance_analysis_full_dataset_{latent_dim}_latent_dim_{output_file_postfix}.csv"


    raw_test_data_filename = input_file
    predictions_test_data_filename = output_file
    json_filename = evaluation_metadata_filename
    difference_binary_value_summary_filename = performance_analysis_binary_file
    difference_real_value_summary_filename = performance_analysis_real_file

    print(f"Input file: {raw_test_data_filename}")
    print(f"Output file: {predictions_test_data_filename}")
    print(f"JSON file: {json_filename}")
    print(f"Binary difference summary file: {difference_binary_value_summary_filename}")
    print(f"Real difference summary file: {difference_real_value_summary_filename}")

    #Open the json file which has relevant information about columns
    with open(json_filename, "r") as f:
        col_list_dict = json.load(f)

    #Read the raw and predicted files
    orig = pd.read_csv(raw_test_data_filename)
    preds = pd.read_csv(predictions_test_data_filename)

    if len(orig) == len(preds):
        print(f"\nNumber of rows in input and output files match: {len(orig)}\n")
    else:
        print(f"\nERROR: Number of rows in input file: {len(orig)} does not match with Number of rows in output file: {len(preds)}\n")


    #Check column count in raw and predicted
    num_cols_satisfied = verify_count_of_all_columns(orig.columns, preds.columns)
    print(f"\nNumber of columns predicted is the same in both original and predicted data: {num_cols_satisfied}\n")

    #Check column ordering in raw and predicted
    col_order_satisfied = verify_column_order_of_all_columns(orig.columns, preds.columns)
    print(f"\nColumns in original and predicted are in the same order: {col_order_satisfied}\n")

    #Set tolerance to 0.9 for real value columns
    tolerance_value = 0.9

    #Print this if you want to see all rows and columns that do not match.
    #print_differences(orig, preds, col_list_dict["real_cols"], tolerance_value)

    print(f"\nPrinting column stats of all real columns which exceed tolerance {tolerance_value}\n")
    print_real_difference_summary(orig, preds, col_list_dict["real_cols"], tolerance_value, difference_real_value_summary_filename, print_all_stats)

    tolerance_value = 0
    print(f"\nPrinting column stats of all binary columns which exceed tolerance {tolerance_value}\n")
    print_binary_difference_summary(orig, preds, col_list_dict["binary_cols"], tolerance_value, difference_binary_value_summary_filename, print_all_stats)
