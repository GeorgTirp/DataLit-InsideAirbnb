import math
import warnings
from typing import Dict, Literal
import pandas as pd
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
import gc
import os
from rtdl_revisiting_models import FTTransformer # From https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/README.md
warnings.resetwarnings()
warnings.simplefilter("ignore", ResourceWarning)
warnings.simplefilter("ignore")



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
            feature_types: dict = None):
        
        logging.info("Initializing FT_Transformer")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Set random seeds in all libraries.
        delu.random.seed(0)

        self.test_size = test_split_size
        self.save_path = save_path
        self.identifier = identifier
        self.Feature_Selection = Feature_Selection
        if hparams is None:
            self.hparams = FTTransformer.get_default_kwargs()
        else:
            self.hparams = hparams
        self.feature_types = feature_types
        self.train_n = None
        self.reg_model = None
        
        self.data, self.n_cont_features, self.cat_cardinalities = self.model_specific_preprocess(
            data_df, 
            Feature_Selection, 
            feature_types)
        del data_df
        self.metrics = None
        self.model = FTTransformer(
            n_cont_features=self.n_cont_features, 
            cat_cardinalities=self.cat_cardinalities, 
            d_out=1, 
            **self.hparams,).to(self.device)
        logging.info("FT_Transformer initialized successfully")
        

    def model_specific_preprocess(
            self, 
            data_df: pd.DataFrame, 
            Feature_Selection: dict = None, 
            feature_types :dict = None, 
            test_size=None) -> Tuple:
        """ Preprocess the data for the TabPFN model"""
        logging.info("Starting model-specific preprocessing")
        
        if Feature_Selection is None:
            Feature_Selection = self.Feature_Selection
        if feature_types is None:
            feature_types = self.feature_types
        if test_size is None:
            test_size = self.test_size

        mask = data_df[Feature_Selection['features'] + [Feature_Selection['target']]].notna().all(axis=1)
        data_df = data_df[mask]
        X = data_df[Feature_Selection['features']]
        y = data_df[Feature_Selection['target']]
        for col in Feature_Selection['features']:
            X[col] = pd.to_numeric(data_df[col], errors='coerce')

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
            all_idx, test_size=test_size, random_state=42
        )
        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=test_size, random_state=42
        )
        self.train_n = len(train_idx)
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
        
        data ={
            part: {k: torch.as_tensor(v, device=self.device) for k, v in data_numpy[part].items()}
            for part in data_numpy
        }

        # Clear memory
        del X
        del data_df
        del data_numpy
        del X_cat
        del X_cont
        gc.collect()
        gc.get_referrers()
        logging.info("Model-specific preprocessing completed")
        return data, n_cont_features, cat_cardinalities
    
    def train(self, batch_size, patience ,n_epochs, optimizer) -> None:
        logging.info("Training started")
        # Training logic here

        def apply_model(batch: Dict[str, Tensor]) -> Tensor:
            if isinstance(self.model, FTTransformer):
                return self.model(batch["x_cont"], batch.get("x_cat")).squeeze(-1)
            else:
                raise RuntimeError(f"Unknown model type: {type(self.model)}")


        loss_fn = (
             F.mse_loss
        )


        @torch.no_grad()
        def train_evaluate(part: str) -> float:
            self.model.eval()

            eval_batch_size = 8096
            y_pred = (
                torch.cat(
                    [
                        apply_model(batch)
                        for batch in delu.iter_batches(self.data[part], eval_batch_size)
                    ]
                )
                .cpu()
                .numpy()
            )
            y_true = self.data[part]["y"].cpu().numpy()

            Y_std = self.data["train"]["y"].std().item()
            score = -(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5 * Y_std)
            return score  # The higher -- the better.


        print(f'Test score before training: {train_evaluate("test"):.4f}')
        
        epoch_size = math.ceil(self.train_n / batch_size)
        timer = delu.tools.Timer()
        early_stopping = delu.tools.EarlyStopping(patience, mode="max")
        best = {
            "val": -math.inf,
            "test": -math.inf,
            "epoch": -1,
        }

        print(f"Device: {self.device.type.upper()}")
        print("-" * 88 + "\n")
        timer.run()
        for epoch in range(n_epochs):
            for batch in tqdm(
                delu.iter_batches(self.data["train"], batch_size, shuffle=True),
                desc=f"Epoch {epoch}",
                total=epoch_size,
            ):
                self.model.train()
                optimizer.zero_grad()
                loss = loss_fn(apply_model(batch), batch["y"])
                loss.backward()
                optimizer.step()

            val_score = train_evaluate("val")
            test_score = train_evaluate("test")
            print(f"(val) {val_score:.4f} (test) {test_score:.4f} [time] {timer}")

            early_stopping.update(val_score)
            if early_stopping.should_stop():
                break
            
            if val_score > best["val"]:
                print("🌸 New best epoch! 🌸")
                best = {"val": val_score, "test": test_score, "epoch": epoch}
            torch.cuda.empty_cache()
            gc.collect()

        print("\n\nResult:")
        print(best)
        logging.info("Training completed")

    def save_model(self, save_path: str) -> None:
        """Save the model"""
        logging.info("Saving model")
        # Saving logic here
        logging.info("Model saved")
        pass

    def predict(self, data_df: pd.DataFrame, save_results=False, evaluate=True) -> Dict:
        """Predict using the trained model"""
        logging.info("Prediction started")
        # Reuse model_specific_preprocess for preprocessing
        data, _, _ = self.model_specific_preprocess(data_df, self.Feature_Selection, self.feature_types, test_size=self.test_size)
        
        self.model.eval()
        with torch.no_grad():
            train_predictions = self.model(data["train"]["x_cont"], data["train"].get("x_cat")).squeeze(-1).cpu().numpy()
        y_train = data["train"]["y"].cpu().numpy()
        with torch.no_grad():
            test_predictions = self.model(data["test"]["x_cont"], data["test"].get("x_cat")).squeeze(-1).cpu().numpy()
        y_test = data["test"]["y"].cpu().numpy()

        predictions = np.concatenate((train_predictions, test_predictions))
        y = np.concatenate((y_train, y_test))
        if save_results == True:
            # Optionally save predictions
            results_df = pd.DataFrame({'y_test': y, 'y_pred': predictions})
            results_df.to_csv(f'{self.save_path}/{self.identifier}_results.csv', index=False)

        if evaluate == True:
            ft_mse = mean_squared_error(y_test, test_predictions)
            ft_r2 = r2_score(y_test, test_predictions)
        
            logging.info(f"TabPFN MSE: {ft_mse}, R2: {ft_r2}")
        
            ft_metrics = {
               'mse': ft_mse,
                'r2': ft_r2,
            }
            self.metrics = ft_metrics

        logging.info("Evaluation completed")
        logging.info("Prediction completed")
        return predictions

        

    def evaluate(self) -> Tuple:
        """ Evaluate the models using mean squared error, r2 score and cross validation"""
        logging.info("Evaluation started")
        X_train, X_test, y_train, y_test = self.train_split

        # Linear Regression
        ft_pred = self.model.predict(X_test)
        ft_mse = mean_squared_error(y_test, ft_pred)
        ft_r2 = r2_score(y_test, ft_pred)
        
        logging.info(f"FT-Transformer MSE: {ft_mse}, R2: {ft_r2}")
        
        ft_metrics = {
            'mse': ft_mse.item(),
            'r2': ft_r2.item(),
        }
        self.metrics = ft_metrics

        logging.info("Evaluation completed")
        return ft_metrics

    def feature_importance(self, save_results=True) -> Dict:
        """Return the feature importance for the Random Forest and linear model"""
        logging.info("Feature importance calculation started")
        # Feature importance logic here
        logging.info("Feature importance calculation completed")
        return {}

    def plot(self):
        pass

if __name__ == "__main__":
    logging.info("Script started")
    #folder_path = "/Users/georgtirpitz/Documents/Data_Literacy"
    folder_path = "/home/georg/Documents/Master/Data_Literacy"
    with open(folder_path + "/city_listings.csv", 'r') as file:
        data_df = pd.read_csv(file)
    safe_path = folder_path + "DataLit-InsideAirbnb" + "/results" + "/ft_test"
    if not os.path.exists(safe_path):
        os.makedirs(safe_path)
    identifier = "ft_transformer"
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
        feature_types)
    
    param_groups = [
    {
        "params": [p for n, p in model.model.named_parameters() if "bias" not in n and "LayerNorm" not in n],
        "weight_decay": 1e-3,
    },
    {
        "params": [p for n, p in model.model.named_parameters() if "bias" in n or "LayerNorm" in n],
        "weight_decay": 0.0,
    },
]

    # Create the AdamW optimizer with the same settings
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=0.00001,
        betas=(0.9, 0.99),
        eps=1e-08,
        amsgrad=False
    )
    model.model_specific_preprocess(data_df, Feature_Selection)
    model.train(batch_size=256, patience=200, n_epochs=3, optimizer=optimizer)
    model.predict(data_df, save_results=True, evaluate=True)
    logging.info(f"Metrics: {model.metrics}")
    #model.save_model(safe_path)
    logging.info("Script finished")