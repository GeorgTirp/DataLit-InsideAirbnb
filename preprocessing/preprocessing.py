import csv
import pandas as pd
import numpy as np
import os



import gzip

def is_gzipped(file_path):
    try:
        with open(file_path, 'rb') as f:
            return f.read(2) == b'\x1f\x8b'  # Check for gzip magic number
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return False

def read(folder_path):
    # Initialize the dictionary to store DataFrames
    data_dict = {}

    for root, dirs, files in os.walk(folder_path):
        # Get the name of the subdirectory
        subdirectory_name = os.path.basename(root)

        # Skip the root folder itself if needed (optional)
        if subdirectory_name == os.path.basename(folder_path):
            continue
        
        # Initialize a list for the current subdirectory
        data_dict[subdirectory_name] = []

        print(f"Processing folder: {root}")

        for file_name in files:
            if file_name.endswith('.csv') or file_name.endswith('.geojson') or file_name.endswith('.csv.gz'):
                file_path = os.path.join(root, file_name)

                try:
                    # If it's a gzipped CSV file, check if it is valid
                    if file_name.endswith('.csv.gz'):
                        if is_gzipped(file_path):
                            print(f"Loading gzipped file: {file_path}")
                            df = pd.read_csv(file_path, compression='gzip')
                        else:
                            print(f"Skipping invalid gzip file: {file_path}")
                            continue
                    elif file_name.endswith('.geojson'):
                        print(f"Loading geojson file: {file_path}")
                        df = pd.read_json(file_path)  # Adjust based on the specific geojson handling
                    else:
                        print(f"Loading regular CSV file: {file_path}")
                        df = pd.read_csv(file_path)

                    # Append the DataFrame to the list for this subdirectory
                    data_dict[subdirectory_name].append(df)

                    print(f"Loaded {file_path} into {subdirectory_name}'s list of DataFrames")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    return data_dict




def first_merge(data_dict):
    for city, df_list in data_dict.items():
        # Concatenate all DataFrames in the list for the current city
        merged_df = pd.concat(df_list, ignore_index=True)

        # Replace the list of DataFrames with the merged DataFrame
        data_dict[city] = merged_df

    # Print the keys and the shape of the merged DataFrames to verify
    for city, df in data_dict.items():
        print(f"{city}: {df.shape}")
    return data_dict



def second_merge(data_dict):
    df_list_with_region = []

    for city, df in data_dict.items():
        df.insert(0, 'region', city)
        df_list_with_region.append(df)

    combined_df = pd.concat(df_list_with_region, ignore_index=True)
    return combined_df

def preprocess_data(data_dict):
    merged_data_dict = first_merge(data_dict)
    combined_df = second_merge(merged_data_dict)
    output_path = 'data/preprocessed_data.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    return combined_df

if __name__ == "__main__":
    folder_path = 'data/example_data'
    data_dict = read(folder_path)
    merged_data_dict = first_merge(data_dict)