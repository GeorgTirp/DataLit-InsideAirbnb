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
from scipy.stats import pearsonr
from rtdl_revisiting_models import FTTransformer # From https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/README.md
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from sklearn.model_selection import KFold
warnings.resetwarnings()
warnings.simplefilter("ignore", ResourceWarning)
warnings.simplefilter("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Wrapper for FTTransformer to handle SHAP input
class WrappedFTTransformerModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device  # Get the model's device

    def forward(self, X):
        """
        SHAP provides `X` as a single tensor, so we split it into X_cont and X_cat.
        """
        if isinstance(X, tuple):
            X_cont, X_cat = X
        else:
            X_cont = X
            X_cat = None  # Handle missing categorical input

        # Ensure tensors are correctly formatted with requires_grad=True
        X_cont = torch.tensor(X_cont, dtype=torch.float32, device=self.device, requires_grad=True)

        if X_cat is not None:
            X_cat = torch.tensor(X_cat, dtype=torch.int64, device=self.device)

        return self.model(x_cont=X_cont, x_cat=X_cat)

# Custom SHAP deep explainer for FTTransformer
class FTTransformerDeepExplainer(shap.explainers._deep.DeepExplainer):
    def __init__(self, model, data, check_additivity):
        assert isinstance(data, tuple) and len(data) == 2, "Data must be a tuple of (X_cont, X_cat)."

        self.device = next(model.parameters()).device  # Get model device
        self.model = WrappedFTTransformerModel(model).to(self.device)  # Wrap model

        # Convert data to tensors and move to device
        self.X_cont, self.X_cat = data
        self.X_cont = self.X_cont.to(self.device).detach().clone().requires_grad_(True)
        self.X_cat = self.X_cat.to(self.device) if self.X_cat is not None else None
        self.check_additivity = check_additivity
        
        # SHAP expects a **single tensor**, so we concatenate
        background_data = self._prepare_data_for_shap(self.X_cont, self.X_cat)

        # Initialize SHAP's DeepExplainer
        super().__init__(self.model, background_data)

    def _prepare_data_for_shap(self, X_cont, X_cat):
        """
        Prepare the background data for SHAP.

        Args:
            X_cont (Tensor): Continuous features.
            X_cat (Tensor or None): Categorical features.

        Returns:
            Tensor: Combined input tensor for SHAP.
        """
        if X_cat is None:
            X_cat = torch.empty((X_cont.shape[0], 0), device=self.device, dtype=torch.int64)

        # SHAP expects a **single tensor input**, so we concatenate
        return torch.cat([X_cont, X_cat], dim=1)

    def shap_values(self, X):
        """
        Override shap_values to ensure X is correctly formatted.
        """
        assert isinstance(X, tuple) and len(X) == 2, "X must be a tuple (X_cont, X_cat)."

        X_cont, X_cat = X
        X_cont = X_cont.to(self.device).detach().clone().requires_grad_(True)
        X_cat = X_cat.to(self.device) if X_cat is not None else None

        # SHAP expects a **single tensor input**, so we concatenate
        X_input = torch.cat([X_cont, X_cat], dim=1)

        return super().shap_values(X_input, check_additivity=self.check_additivity)

class FTTransformerKernelExplainer:
    def __init__(self, model, background_data, nsamples=1000):
        """
        Initialize SHAP KernelExplainer for FT-Transformer.
        
        Args:
        - model: The trained FT-Transformer model.
        - background_data: Tuple (X_cont, X_cat) for Kernel SHAP sampling.
        - nsamples: Number of samples for SHAP estimation.
        """
        self.model = model
        self.model.eval()  # Ensure evaluation mode
        
        # Store background data (use a larger sample for better SHAP results)
        X_cont_bg, X_cat_bg = background_data
        self.background_data = np.concatenate([X_cont_bg[:500], X_cat_bg[:500]], axis=1)

        # Define KernelExplainer
        self.explainer = shap.KernelExplainer(self._predict_fn, self.background_data)
        self.nsamples = nsamples

    def _predict_fn(self, X_np):
        """
        Model prediction function for KernelExplainer.
        Converts NumPy array back into correct FT-Transformer input format.
        """
        # Split back into continuous and categorical features
        num_cont_features = X_np.shape[1] // 2  # Assume equal split for now
        X_cont = torch.tensor(X_np[:, :num_cont_features], dtype=torch.float32)
        X_cat = torch.tensor(X_np[:, num_cont_features:], dtype=torch.int64)

        with torch.no_grad():
            preds = self.model(X_cont, X_cat).cpu().numpy()

        return preds.flatten()  # Flatten to match SHAP expectations

    def shap_values(self, X):
        """
        Compute SHAP values using KernelExplainer.
        
        Args:
        - X: Tuple (X_cont, X_cat) of input samples.
        
        Returns:
        - SHAP values for each feature.
        """
        # Convert tuple into a single NumPy array before passing to Kernel SHAP
        X_np = np.concatenate([X[0], X[1]], axis=1)
        return self.explainer.shap_values(X_np, nsamples=self.nsamples)

     


# FT TRansformer Regressor on Panda Dataframes with feature importance
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
        # Mask to filter rows 
        mask_y = y <= 1000

        # Apply the mask to filter y and drop corresponding rows in X
        X = X[mask_y]
        y = y[mask_y]
        
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
        del data_numpy
        del X_cat
        del X_cont
        gc.collect()
        gc.get_referrers()
        logging.info("Model-specific preprocessing completed")
        return data, n_cont_features, cat_cardinalities
    
    def train(self, batch_size, patience ,n_epochs, optimizer, scheduler=None, second_order_method=False) -> None:
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
                if second_order_method:
                    def closure():
                        self.model.train()
                        optimizer.zero_grad()
                        loss = loss_fn(apply_model(batch), batch["y"])
                        loss.backward()
                        return loss
                    optimizer.step(closure)
                
                else:
                    self.model.train()
                    optimizer.zero_grad()
                    loss = loss_fn(apply_model(batch), batch["y"])
                    loss.backward()
                    optimizer.step()
                if scheduler != None:
                    scheduler.step()

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
        """Save only the model parameters."""
        logging.info(f"Saving model parameters to {save_path}")
    
        try:
            torch.save(self.model.state_dict(), save_path)
            logging.info("Model parameters saved successfully.")
        except Exception as e:
            logging.error(f"Error saving model parameters: {e}")
        

    def predict(self, data_df: pd.DataFrame, save_results=False, evaluate=True, folds=5) -> Dict:
        """Predict using the trained model"""
        gc.collect()  # Run Python's garbage collector
        torch.cuda.empty_cache()  # Clears unused GPU memory
        logging.info("Prediction started")
        # Reuse model_specific_preprocess for preprocessing
        data, _, _ = self.model_specific_preprocess(data_df, self.Feature_Selection, self.feature_types, test_size=self.test_size)
        
        self.model.eval()
        eval_batch_size = 1024  # Reduce batch size to avoid memory overflow
    
        def batch_predict(data_part):
            """Helper function to perform batch-wise prediction"""
            y_preds = []
            with torch.no_grad():
                for batch in delu.iter_batches(data[data_part], eval_batch_size):
                    batch_pred = self.model(batch["x_cont"], batch.get("x_cat")).squeeze(-1).cpu().numpy()
                    y_preds.append(batch_pred)
            return np.concatenate(y_preds)

        # Generate predictions in batches
        train_predictions = batch_predict("train")
        test_predictions = batch_predict("test")

        y_train = data["train"]["y"].cpu().numpy()
        y_test = data["test"]["y"].cpu().numpy()

        predictions = np.concatenate((train_predictions, test_predictions))
        y = np.concatenate((y_train, y_test))
        if save_results == True:
            # Optionally save predictions
            results_df = pd.DataFrame({'y_test': y, 'y_pred': predictions})
            results_df.to_csv(f'{self.save_path}/{self.identifier}_results.csv', index=False)

        if evaluate == True:
            
            mse = mean_squared_error(y_test, test_predictions)
            r, p_price = pearsonr(y_test, test_predictions)
            r2 = r**2
            logging.info(f"{self.identifier} MSE: {mse}, R2: {r2}")
            logging.info(f"{self.identifier} P_value: {p_price}")

            metrics = {
                'mse': mse,
                'r2': r2,
                'p_value': p_price
            }
            self.metrics = metrics
            self.metrics = metrics

        logging.info("Evaluation completed")
        logging.info("Prediction completed")
        return predictions


    def feature_importance(self, save_results=True) -> Dict:
        """Return the feature importance using SHAP values."""
        logging.info("Starting SHAP importance evaluation...")
        shap.initjs()  

        # Set model to eval so that Batchnorm and Dropout layers do not interfere with gradients from explainer
        self.model.eval()
        
        # Extract preprocessed features
        X_cont = self.data["train"]["x_cont"].to(self.device)
        X_cat = self.data["train"].get("x_cat", None)

        # Ensure categorical tensor is valid
        if X_cat is None:
            X_cat = torch.empty((X_cont.shape[0], 0), device=self.device, dtype=torch.int64)
        
        # Prepare background data
        background_data = (X_cont[:50], X_cat[:50])

        # Prepare evaluation data
        eval_data = (X_cont[:500], X_cat[:500])

        # Convert `eval_data` to a list to avoid tuple issues
        eval_data = [tensor for tensor in eval_data]

        # Initialize SHAP explainer
        explainer = FTTransformerDeepExplainer(self.model, background_data, check_additivity=False)
        #explainer = FTTransformerKernelExplainer(self.model, background_data)
        # Compute SHAP values
        shap_values = explainer.shap_values((X_cont[:500], X_cat[:500]))

        # Prepare feature names
        feature_names = self.feature_types.get("continuous", []) + self.feature_types.get("categorical", [])

        # Convert tensors to NumPy
        eval_cont_np = eval_data[0].cpu().numpy()
        eval_cat_np = eval_data[1].cpu().numpy().astype(int) if eval_data[1].shape[1] > 0 else None
        eval_combined = np.concatenate([eval_cont_np, eval_cat_np], axis=1) if eval_cat_np is not None else eval_cont_np

        # Generate SHAP summary plots
        plt.figure()
        shap.summary_plot(shap_values, eval_combined, feature_names=feature_names, show=False, max_display=40)
        plt.title(f'{self.identifier} SHAP Summary Plot')
        if save_results:
            plt.savefig(f'{self.save_path}/{self.identifier}_shap_summary.png')
            plt.close()

        plt.figure()
        shap.summary_plot(shap_values, eval_combined, plot_type="bar", feature_names=feature_names, show=False)
        plt.title(f'{self.identifier} SHAP Feature Importance')
        if save_results:
            plt.savefig(f'{self.save_path}/{self.identifier}_shap_bar.png')
            plt.close()
            
        logging.info("Finished feature importance evaluation.")
        return shap_values

    def plot(self):
        """ Plot """
        results_df = pd.read_csv(f'{self.save_path}/{self.identifier}_results.csv')
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['y_test'], results_df['y_pred'], alpha=0.5)
        plt.plot([results_df['y_test'].min(), results_df['y_test'].max()], 
                 [results_df['y_test'].min(), results_df['y_test'].max()], 
                 color='red', linestyle='--', linewidth=2)
        plt.text(results_df['y_test'].min(), 
                results_df['y_pred'].max(), 
                f'R^2: {self.metrics["r2"]:.2f}\nP-value: {self.metrics["p_value"]:.2e}', 
                fontsize=12, 
                verticalalignment='top', 
                bbox=dict(facecolor='white', 
                alpha=0.5))
        plt.xlabel('Actual prices')
        plt.ylabel('Predicted prices')
        plt.title(f'Actual vs Predicted prices ({self.identifier})')
        plt.grid(True)
        plt.savefig(f'{self.save_path}/{self.identifier}_actual_vs_predicted.png')
        plt.show()
        plt.close()
