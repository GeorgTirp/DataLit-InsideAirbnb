import math
import warnings
from typing import Dict, Literal
import pandas as pd
warnings.simplefilter("ignore")
import delu  # Deep Learning Utilities: https://github.com/Yura52/delu
import numpy as np
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from tqdm.std import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from typing import Tuple, Dict
import shap
from add_custom_features import AddCustomFeatures
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from rtdl_revisiting_models import FTTransformer # From https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/README.md
warnings.resetwarnings()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class FT_Transfomer():
    """ Fit, evaluate, and get attributions regression models (current: Random Forest and Linear Regression)"""
    def __init__(
            self,
            data_df: pd.DataFrame, 
            Feature_Selection: dict, 
            test_split_size:float = 0.2,
            save_path: str = None,
            identifier: str = None,
            hparams: dict = None,
            feature_types: dict = None,
            optimizer: torch.optim.Optimizer = None):
        
        logging.info("Initializing FT_Transformer")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Set random seeds in all libraries.
        delu.random.seed(0)
        self.save_path = save_path
        self.identifier = identifier
        self.Feature_Selection = Feature_Selection
        if hparams is None:
            self.hparams = FTTransformer.get_default_kwargs()
        else:
            self.hparams = hparams
        self.feature_types = feature_types
        
        self.reg_model = None
        self.X, self.y, self.data_numpy, self.n_cont_features, self.cat_cardinalities = self.model_specific_preprocess(
            data_df, 
            Feature_Selection, 
            feature_types)
        self.train_split = train_test_split(self.X, self.y, test_size=test_split_size, random_state=42)
        self.metrics = None
        self.model = FTTransformer(
            n_cont_features=self.n_cont_features, 
            cat_cardinalities=self.cat_cardinalities, 
            d_out=1, 
            **self.hparams,).to(self.device)
        if optimizer is None:
            self.optimizer = self.model.make_default_optimizer()
        else:
            self.optimizer = optimizer
        logging.info("FT_Transformer initialized successfully")

    def model_specific_preprocess(self, data_df: pd.DataFrame, Feature_Selection: dict = None, feature_types :dict = None) -> Tuple:
        """ Preprocess the data for the TabPFN model"""
        logging.info("Starting model-specific preprocessing")
        # Ensure all features are numeric
        if Feature_Selection is None:
            Feature_Selection = self.Feature_Selection
        if feature_types is None:
            feature_types = self.feature_types
        data_df = data_df.dropna(subset=Feature_Selection['features'] + [Feature_Selection['target']])
        X = data_df[Feature_Selection['features']]
        y = data_df[Feature_Selection['target']]
        X = X.apply(pd.to_numeric, errors='coerce')
        # Remove dollar sign and convert to float
        if y.dtype == object:
            y = y.replace('[\$,]', '', regex=True).astype(float)
        y = y.astype(np.float32).to_numpy()

        # >>> Continuous features.
        continuous_features = feature_types["continuous"]
        categorical = feature_types["categorical"]
        n_cont_features = len(continuous_features)
        X_cont: np.ndarray= X[continuous_features].astype(np.float32).to_numpy()
        n_cont_features = X_cont.shape[1]
        
        # Categorical features
        if categorical:
            X_cat = X[categorical].apply(lambda x: pd.factorize(x)[0]).values
            cat_cardinalities = [X[cat].nunique() for cat in categorical]
        else:
            X_cat = None
            cat_cardinalities = []
        
        # >>> Split the dataset.
        all_idx = np.arange(len(y))
        trainval_idx, test_idx = train_test_split(
            all_idx, train_size=0.8, random_state=42
        )
        train_idx, val_idx = train_test_split(
            trainval_idx, train_size=0.8
        )
        data_numpy = {
            "train": {"x_cont": X_cont[train_idx], "y": y[train_idx]},
            "val": {"x_cont": X_cont[val_idx], "y": y[val_idx]},
            "test": {"x_cont": X_cont[test_idx], "y": y[test_idx]},
        }
        if X_cat is not None:
            data_numpy["train"]["x_cat"] = X_cat[train_idx]
            data_numpy["val"]["x_cat"] = X_cat[val_idx]
            data_numpy["test"]["x_cat"] = X_cat[test_idx]

        #Fancy preprocessing strategy.
        # The noise is added to improve the output of QuantileTransformer in some cases.
        X_cont_train_numpy = data_numpy["train"]["x_cont"]
        noise = (
            np.random.default_rng(0)
            .normal(0.0, 1e-5, X_cont_train_numpy.shape)
            .astype(X_cont_train_numpy.dtype)
        )
        preprocessing = sklearn.preprocessing.QuantileTransformer(
            n_quantiles=max(min(len(train_idx) // 30, 1000), 10),
            output_distribution="normal",
            subsample=10**9,
        ).fit(X_cont_train_numpy + noise)
        del X_cont_train_numpy

        for part in data_numpy:
            data_numpy[part]["x_cont"] = preprocessing.transform(data_numpy[part]["x_cont"])

        # >>> Label preprocessing.
        Y_mean = data_numpy["train"]["y"].mean().item()
        Y_std = data_numpy["train"]["y"].std().item()
        for part in data_numpy:
            data_numpy[part]["y"] = (data_numpy[part]["y"] - Y_mean) / Y_std

        logging.info("Model-specific preprocessing completed")
        return X, y, data_numpy, n_cont_features, cat_cardinalities
    
    def train(self, batchsize, epochs) -> None:
        logging.info("Training started")
        # Training logic here
        logging.info("Training completed")

    def predict(self, X_in: pd.DataFrame, save_results=False) -> Dict:
        """Predict using the trained model"""
        logging.info("Prediction started")
        # Prediction logic here
        logging.info("Prediction completed")
        return {}

    def evaluate(self) -> Tuple:
        """ Evaluate the models using mean squared error, r2 score and cross validation"""
        logging.info("Evaluation started")
        X_train, X_test, y_train, y_test = self.train_split

        # Linear Regression
        reg_pred = self.reg_model.predict(X_test)
        reg_mse = mean_squared_error(y_test, reg_pred)
        reg_r2 = r2_score(y_test, reg_pred)
        
        logging.info(f"TabPFN MSE: {reg_mse}, R2: {reg_r2}")
        
        tabpfn_metrics = {
            'mse': reg_mse,
            'r2': reg_r2,
        }
        self.metrics = tabpfn_metrics

        logging.info("Evaluation completed")
        return tabpfn_metrics

    def feature_importance(self, save_results=True) -> Dict:
        """Return the feature importance for the Random Forest and linear model"""
        logging.info("Feature importance calculation started")
        # Feature importance logic here
        logging.info("Feature importance calculation completed")
        return {}

    def plot(self):
        """ Plot """
        logging.info("Plotting started")
        # Plotting logic here
        logging.info("Plotting completed")

if __name__ == "__main__":
    logging.info("Script started")
    folder_path = "/Users/georgtirpitz/Documents/Data_Literacy"
    data_df = pd.read_csv(folder_path + "/city_listings.csv")
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
    hparams = {
        "n_blocks": 3,
        "d_block": 192,
        "attention_n_heads": 8,
        "attention_dropout": 0.2,
        "ffn_d_hidden": None,
        "ffn_d_hidden_multiplier": 4 / 3,
        "ffn_dropout": 0.1,
        "residual_dropout": 0.0}
    
    feature_types = {
        "continuous": [
            "review_scores_value",
            "distance_to_city_center",
            "average_review_length"],
        "categorical": [
            "accommodates",
            "bathrooms",
            "bedrooms",
            "beds"
        ]
    }
    optimizer = None #so that it makes the default optimizer in the init
    
    model = FT_Transfomer(
        data_df, 
        Feature_Selection,
        test_split_size,
        safe_path,
        identifier,
        hparams,
        feature_types,
        optimizer)
    model.model_specific_preprocess(data_df, Feature_Selection)
    
    logging.info("Script finished")
    