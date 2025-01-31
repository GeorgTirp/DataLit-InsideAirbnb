import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
import shap
import logging
from tqdm import tqdm
from add_custom_features import AddCustomFeatures
import shap
import logging
from tqdm import tqdm
from scipy.stats import pearsonr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaseRegressionModel:
    """ Base class for regression models """
    def __init__(
            self,
            data_df: pd.DataFrame, 
            feature_selection: dict, 
            test_split_size: float = 0.2,
            save_path: str = None,
            identifier: str = None,
            top_n: int = 10):
        
        logging.info("Initializing BaseRegressionModel class...")
        self.feature_selection = feature_selection
        self.top_n = top_n
        self.X, self.y = self.model_specific_preprocess(data_df)
        self.train_split = train_test_split(self.X, self.y, test_size=test_split_size, random_state=42)
        self.save_path = save_path
        self.identifier = identifier
        self.metrics = None

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logging.info("Finished initializing BaseRegressionModel class.")

    def model_specific_preprocess(self, data_df: pd.DataFrame) -> Tuple:
        """ Preprocess the data for the model"""
        logging.info("Starting model-specific preprocessing...")
        data_df = data_df.dropna(subset=self.feature_selection['features'] + [self.feature_selection['target']])
        X = data_df[self.feature_selection['features']]
        y = data_df[self.feature_selection['target']]
        X = X.fillna(X.mean())
        X = X.apply(pd.to_numeric, errors='coerce')
        if y.dtype == object:
            y = y.replace('[\$,]', '', regex=True).astype(float)
        logging.info("Finished model-specific preprocessing.")
        # Z-score normalization for numerical features
        numerical_features = X.select_dtypes(include=[np.number]).columns
        X[numerical_features] = X[numerical_features].apply(stats.zscore)
        return X, y

    def predict(self, X: pd.DataFrame, y: pd.DataFrame = None, save_results=False) -> np.ndarray:
        """ Predict using the trained model"""
        logging.info("Starting prediction...")
        nan_counts = X.isna().sum()
        logging.info("NaN counts per column:")
        logging.info(nan_counts)
        pred = self.model.predict(X)
        
        if save_results:
            results_df = pd.DataFrame({'y_test': y, 'y_pred': pred})
            results_df.to_csv(f'{self.save_path}/{self.identifier}_results.csv', index=False)
        logging.info("Finished prediction.")
        return pred

    def evaluate(self) -> Dict:
        """ Evaluate the model using mean squared error, r2 score and cross validation"""
        logging.info("Starting model evaluation...")
        X_train, X_test, y_train, y_test = self.train_split

        pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        r2, p_price = pearsonr(y_test, pred)

        logging.info(f"{self.identifier} MSE: {mse}, R2: {r2}")
        logging.info(f"{self.identifier} P_value: {p_price}")

        metrics = {
            'mse': mse,
            'r2': r2,
            'p_value': p_price
        }
        self.metrics = metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f'{self.save_path}/{self.identifier}_metrics.csv', index=False)
        logging.info("Finished model evaluation.")
        return metrics

    def feature_importance(self, top_n: int = 10, save_results=True) -> Dict:
        """ Return the feature importance for the model"""
        logging.info("Starting feature importance evaluation...")
        if hasattr(self.model, 'feature_importances_'):
            attribution = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            attribution = abs(self.model.coef_) / sum(abs(self.model.coef_))
        else:
            raise AttributeError("Model does not have feature_importances_ or coef_ attribute")

        feature_names = self.feature_selection['features']
        indices = attribution.argsort()[-top_n:][::-1]
        top_features = {feature_names[i]: attribution[i] for i in indices}

        if save_results:
            np.save(f'{self.save_path}/{self.identifier}_feature_importance.npy', top_features)
        
        self.importances = top_features

        logging.info("Starting SHAP importance evaluation...")
        shap.initjs()
        background_data = self.X.sample(50, random_state=42)
        
        
        if hasattr(self.model, 'feature_importances_'):
            explainer = shap.TreeExplainer(self.model)
            shap_values = []
            for row in tqdm(self.X.itertuples(index=False), total=len(self.X), desc="Computing SHAP values"):
                row_shap = explainer.shap_values(pd.DataFrame([row], columns=self.X.columns))
                shap_values.append(row_shap)
            # Convert list of arrays to a single array
            shap_values = np.array(shap_values).squeeze()
        else:
            explainer = shap.LinearExplainer(self.model, background_data)
            shap_values = explainer.shap_values(self.X)
        
        # Plot aggregated SHAP values (Feature impact)
        shap.summary_plot(shap_values, features=self.X, feature_names=self.X.columns, show=False, max_display=40)
        plt.title(f'{self.identifier} SHAP Summary Plot (Aggregated)', fontsize=16)
        if save_results:
            plt.subplots_adjust(top=0.90)
            plt.savefig(f'{self.save_path}/{self.identifier}_shap_aggregated_beeswarm.png')
            plt.close()
            shap.summary_plot(shap_values, self.X, plot_type="bar", show=False)
            plt.savefig(f'{self.save_path}/{self.identifier}_shap_aggregated_bar.png')
            plt.close()
        

        logging.info("Finished feature importance evaluation.")
        return top_features

    def plot(self) -> None:
        """ Plot feature importances"""
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

class LinearRegressionModel(BaseRegressionModel):
    """ Linear Regression Model """
    def __init__(
            self,
            data_df: pd.DataFrame, 
            feature_selection: dict, 
            test_split_size: float = 0.2,
            save_path: str = None,
            identifier: str = None,
            top_n: int = 10):
        
        super().__init__(data_df, feature_selection, test_split_size, save_path, identifier, top_n)
        self.model = LinearRegression()

    def fit(self) -> None:
        """ Train the Linear Regression model"""
        logging.info("Starting Linear Regression model training...")
        X_train, X_test, y_train, y_test = self.train_split
        self.model.fit(X_train, y_train)
        logging.info("Finished Linear Regression model training.")

class RandomForestModel(BaseRegressionModel):
    """ Random Forest Model """
    def __init__(
            self,
            data_df: pd.DataFrame, 
            feature_selection: dict, 
            rf_hparams: dict, 
            test_split_size: float = 0.2,
            save_path: str = None,
            identifier: str = None,
            top_n: int = 10):
        
        super().__init__(data_df, feature_selection, test_split_size, save_path, identifier, top_n)
        self.rf_hparams = rf_hparams
        self.model = RandomForestRegressor(**self.rf_hparams)

    def fit(self) -> None:
        """ Train the Random Forest model"""
        logging.info("Starting Random Forest model training...")
        X_train, X_test, y_train, y_test = self.train_split
        self.model.fit(X_train, y_train)
        logging.info("Finished Random Forest model training.")

if __name__ == "__main__":
    logging.info("Starting main execution...")
    folder_path = "/home/georg/Documents/Master/Data_Literacy"
    data_df = pd.read_csv(folder_path + "/city_listings.csv")
    identifier_linear = "test_LinearRegression"
    identifier_rf = "test_RandomForest"
    safe_path_linear = folder_path + "/DataLit-InsideAirbnb" + "/results/" + identifier_linear + "/"
    safe_path_rf = folder_path + "/DataLit-InsideAirbnb" + "/results/" + identifier_rf + "/"
    
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

    test_split_size = 0.2
    add_custom_features = ['distance_to_city_center', 'average_review_length']
    Feature_Adder = AddCustomFeatures(data_df, add_custom_features)
    data_df = Feature_Adder.return_data()
    RandomForest_Hparams = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    }
    n_top_features = 15

    # Linear Regression Model
    linear_model = LinearRegressionModel(
        data_df, 
        Feature_Selection, 
        test_split_size, 
        safe_path_linear, 
        identifier_linear, 
        n_top_features)
    linear_model.fit()
    linear_preds = linear_model.predict(linear_model.train_split[1], linear_model.train_split[3], save_results=True)
    linear_metrics = linear_model.evaluate()
    linear_importances = linear_model.feature_importance(10)
    linear_model.plot()

    # Random Forest Model
    rf_model = RandomForestModel(
        data_df, 
        Feature_Selection, 
        RandomForest_Hparams, 
        test_split_size, 
        safe_path_rf, 
        identifier_rf, 
        n_top_features)
    rf_model.fit()
    rf_preds = rf_model.predict(rf_model.train_split[1], rf_model.train_split[3], save_results=True)
    rf_metrics = rf_model.evaluate()
    rf_importances = rf_model.feature_importance(10)
    rf_model.plot()

    logging.info("Finished main execution.")