from TabPFN_pipeline import TabPFNRegressor
import pandas as pd
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
import logging
import os
from add_custom_features import AddCustomFeatures
from TabPFN_pipeline import TabPFNRegression

def tab_pfn_main(folder_path,  file_path ,safe_path: str, identifier: str):
    #folder_path = "/Users/georgtirpitz/Documents/Data_Literacy"
    data_df = pd.read_csv(file_path)
    safe_path = folder_path + "/results" + "test"
    identifier = "tabpfn"
    # Setting features and target
    Feature_Selection = {
        'features': [
            "accommodates",
            "bathrooms",
            "bedrooms",
            "beds",
            "review_scores_value",
            "distance_to_city_center",
            "average_review_length"],

        'target': 'price'
    }

    # Setting test split size
    test_split_size= 0.2
    add_custom_features = ['distance_to_city_center', 'average_review_length']
    Feature_Adder = AddCustomFeatures(data_df, add_custom_features)
    data_df = Feature_Adder.return_data()
    model = TabPFNRegression(data_df, Feature_Selection, test_split_size, safe_path, identifier)
    X, y = model.model_specific_preprocess(data_df, Feature_Selection)
    model.fit()
    model.predict(X, y, save_results=True)
    model.evaluate()
    model.feature_importance()

    if __name__ == "__main__":
        folder_path = "/home/georg/Documents/Master/Data_Literacy"
        
        for root, dirs, files in os.walk(folder_path):
            for dir_name in dirs:
                subfolder_path = os.path.join(root, dir_name)
                print(f"Processing subfolder: {subfolder_path}")
                # Add your processing code here
                file_path = "/home/georg/Documents/Master/Data_Literacy/city_listings.csv"
                safe_path = "/home/georg/Documents/Master/Data_Literacy/results"
                identifier = "tabpfn"
                tab_pfn_main(folder_path, safe_path, identifier)

