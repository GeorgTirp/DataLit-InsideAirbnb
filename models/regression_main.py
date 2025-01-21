from TabPFN_pipeline import TabPFNRegressor
import pandas as pd
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
import logging
import os
from add_custom_features import AddCustomFeatures
from regression import RegressionModels

def regression_main(folder_path,  file_path ,safe_path: str, identifier: str):
    #folder_path = "/Users/georgtirpitz/Documents/Data_Literacy"
    data_df = pd.read_csv(file_path)
    safe_path = folder_path + "/results" + "test"
    identifier = "tabpfn"
    #folder_path = "/Users/georgtirpitz/Documents/Data_Literacy"
    folder_path = "/home/georg/Documents/Master/Data_Literacy"
    data_df = pd.read_csv(folder_path + "/city_listings.csv")
    safe_path = folder_path + "/results" + "test"
    identifier = ["RandomForest", "LinearRegression"]
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
    RandomForest_Hparams = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    }
    n_top_features = 15 # Number of top features to show

    model = model = RegressionModels(
        data_df, 
        RandomForest_Hparams, 
        Feature_Selection, 
        test_split_size, 
        safe_path, 
        identifier, 
        n_top_features)
    X, y = model.model_specific_preprocess(data_df, Feature_Selection)
    # Setting Hyperparameters and features
    
    model.fit()
    X, y = model.model_specific_preprocess(data_df, Feature_Selection)
    preds = model.predict(X, y, save_results=True)
    metrics = model.evaluate()
    importances = model.feature_importance(10)
    model.plot()
    print("Finished")

    if __name__ == "__main__":
        folder_path = "/home/georg/Documents/Master/Data_Literacy"
        
        for root, dirs, files in os.walk(folder_path):
            for dir_name in dirs:
                subfolder_path = os.path.join(root, dir_name)
                print(f"Processing subfolder: {subfolder_path}")
                # Add your processing code here
                city_name = ""
                file_path = "/home/georg/Documents/Master/Data_Literacy/city_listings.csv"
                safe_path = "/home/georg/Documents/Master/Data_Literacy/results" + city_name
                identifier = "regression"
                regression_main(folder_path, safe_path, identifier)

