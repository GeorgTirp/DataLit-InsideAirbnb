import csv
import pandas as pd
import numpy as np
import os



### Read all CSV files in a folder into a dictionary of DataFrames for every city/region in the folder###


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

                # Read the file into a DataFrame
                if file_name.endswith('.geojson'):
                    df = pd.read_json(file_path)  # Adjust based on the specific geojson handling
                else:
                    df = pd.read_csv(file_path)

                # Append the DataFrame to the list for this subdirectory
                data_dict[subdirectory_name].append(df)

                print(f"Loaded {file_path} into {subdirectory_name}'s list of DataFrames")
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


if __name__ == "__main__":
    folder_path = 'data/example_data'
    data_dict = read(folder_path)
    merged_data_dict = first_merge(data_dict)