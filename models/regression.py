import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
#from preprocessing.preprocessing_test import preprocess_data
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
from add_custom_features import AddCustomFeatures
import shap
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RegressionModels:
    """ Fit, evaluate, and get attributions regression models (current: Random Forest and Linear Regression)"""
    def __init__(
            self,
            data_df: pd.DataFrame, 
            rf_hparams: dict, 
            Feature_Selection: dict, 
            test_split_size:float = 0.2,
            save_path: str = None,
            identifier: str = None,
            top_n: int = 10):
        
        logging.info("Initializing RegressionModels class...")
        self.Feature_Selection = Feature_Selection
        self.top_n = top_n
        self.rf_hparams = rf_hparams
        self.linear_model = None
        self.rf_model = None
        X, y = self.model_specific_preprocess(data_df)
        self.train_split = train_test_split(X, y, test_size=test_split_size, random_state=42)
        self.save_path = save_path
        self.identifier_rf, self.identifier_linear = identifier
        self.linear_importances = None
        self.rf_importances = None

        save_path = f'{save_path}' 
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logging.info("Finished initializing RegressionModels class.")

    def model_specific_preprocess(self, data_df: pd.DataFrame, Feature_Selection: dict = None) -> Tuple:
        """ Preprocess the data for the TabPFN model"""
        logging.info("Starting model-specific preprocessing...")
        # Ensure all features are numeric
        if Feature_Selection is None:
            Feature_Selection = self.Feature_Selection
        # Ensure all features are numeric
        data_df = data_df.dropna(subset=Feature_Selection['features'] + [Feature_Selection['target']])
        X = data_df[Feature_Selection['features']]
        y = data_df[Feature_Selection['target']]
        X = X.fillna(X.mean())
        X = X.apply(pd.to_numeric, errors='coerce')
        # Remove dollar sign and convert to float
        if y.dtype == object:
            y = y.replace('[\$,]', '', regex=True).astype(float)
        logging.info("Finished model-specific preprocessing.")
        return X, y
    
    def fit(self) -> None:
        """ Train and predict using Linear Regression and Random Forest"""
        logging.info("Starting model training...")
        X_train, X_test, y_train, y_test = self.train_split

        # Linear Regression
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        rf_model = RandomForestRegressor(**self.rf_hparams)
        rf_model.fit(X_train, y_train)

        self.linear_model = linear_model
        self.rf_model = rf_model
        logging.info("Finished model training.")
       
    def predict(self, X: pd.DataFrame , y: pd.DataFrame = None, save_results=False) -> Tuple:
        """ Predict using the trained models (for class extern usage)"""
        logging.info("Starting prediction...")
        nan_counts = X.isna().sum()
        logging.info("NaN counts per column:")
        logging.info(nan_counts)
        rf_pred = self.rf_model.predict(X)
        linear_pred = self.linear_model.predict(X)
        
        if save_results:
            # Optionally save predictions
            results_df = pd.DataFrame({'y_test': y, 'y_pred': rf_pred})
            results_df.to_csv(f'{self.save_path}/{self.identifier_rf}_results.csv', index=False)
            results_df = pd.DataFrame({'y_test': y, 'y_pred':linear_pred})
            results_df.to_csv(f'{self.save_path}/{self.identifier_linear}_results.csv', index=False)
        logging.info("Finished prediction.")
        return rf_pred, linear_pred

    def evaluate(self) -> Tuple:
        """ Evaluate the models using mean squared error, r2 score and cross validation"""
        logging.info("Starting model evaluation...")
        X_train, X_test, y_train, y_test = self.train_split

        # Linear Regression
        rf_pred = self.rf_model.predict(X_test)
        linear_pred = self.linear_model.predict(X_test)

        rf_mse = mean_squared_error(y_test, rf_pred)
        linear_mse = mean_squared_error(y_test, linear_pred)

        rf_r2 = r2_score(y_test, rf_pred)
        linear_r2 = r2_score(y_test, linear_pred)

        # Cross-validation for Random Forest
        rf_cv_scores = cross_val_score(self.rf_model, X_train, y_train, cv=10, scoring='r2')
        rf_cv_mean_score = rf_cv_scores.mean()

        # Cross-validation for Linear Regression
        linear_cv_scores = cross_val_score(self.linear_model, X_train, y_train, cv=10, scoring='r2')
        linear_cv_mean_score = linear_cv_scores.mean()

        logging.info(f"Random Forest MSE: {rf_mse}, R2: {rf_r2}")
        logging.info(f"Linear Regression MSE: {linear_mse}, R2: {linear_r2}")
        logging.info(f"Random Forest 10-fold CV Mean R2 Score: {rf_cv_mean_score}")
        logging.info(f"Linear Regression 10-fold CV Mean R2 Score: {linear_cv_mean_score}")

        # Compute p-values for Linear Regression
        params = np.append(self.linear_model.intercept_, self.linear_model.coef_)
        predictions = self.linear_model.predict(X_train)
        newX = np.append(np.ones((len(X_train), 1)), X_train, axis=1)
        MSE = (sum((y_train - predictions) ** 2)) / (len(newX) - len(newX[0]))
        var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
        sd_b = np.sqrt(var_b)
        ts_b = params / sd_b
        linear_p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]

        # Compute Pearson correlation coefficient and p-values for Random Forest
        rf_p_values = []
        for i in range(X_train.shape[1]):
            _ , p = stats.pearsonr(X_test.iloc[:, i], rf_pred)
            rf_p_values.append(p)

        logging.info(f"Random Forest p-values: {rf_p_values}")
        logging.info(f"Linear Regression p-values: {linear_p_values}")

        linear_metrics = {
            'mse': linear_mse,
            'r2': linear_r2,
            'cv_mean_score': linear_cv_mean_score,
            'p_values': linear_p_values
        }

        rf_metrics = {
            'mse': rf_mse,
            'r2': rf_r2,
            'cv_mean_score': rf_cv_mean_score,
            'p_values': rf_p_values
        }

        # Save metrics to CSV
        linear_metrics_df = pd.DataFrame([linear_metrics])
        rf_metrics_df = pd.DataFrame([rf_metrics])

        linear_metrics_df.to_csv(f'{self.save_path}/{self.identifier_linear}_metrics.csv', index=False)
        rf_metrics_df.to_csv(f'{self.save_path}/{self.identifier_rf}_metrics.csv', index=False)
        logging.info("Finished model evaluation.")
        return linear_metrics, rf_metrics
    
    def feature_importance(self, top_n: int = 10, save_results = True) -> Tuple[Dict, Dict]:
        """ Return the feature importance for the Random Forest and linear model"""
        logging.info("Starting feature importance evaluation...")
        # Get attributions
        rf_attribution = self.rf_model.feature_importances_
        linear_attribution = abs(self.linear_model.coef_) / sum(abs(self.linear_model.coef_))
        feature_names = self.Feature_Selection['features']
        rf_indices = rf_attribution.argsort()[-top_n:][::-1]
        linear_indices = linear_attribution.argsort()[-top_n:][::-1]

        top_n = self.top_n
        # Get top n features
        rf_top_features = {feature_names[i]: rf_attribution[i] for i in rf_indices}
        linear_top_features = {feature_names[i]: linear_attribution[i] for i in linear_indices}

        if save_results:
            np.save(f'{self.save_path}/{self.identifier_rf}_feature_importance.npy', rf_top_features)
            np.save(f'{self.save_path}/{self.identifier_linear}_feature_importance.npy', linear_top_features)
        
        self.rf_importances = rf_top_features
        self.linear_importances = linear_top_features

        logging.info("Starting SHAP importance evaluation...")

        # Initialize SHAP
        shap.initjs()

        # Sample background data
        X_train, X_test, y_train, y_test = self.train_split
        background_data = X_train.sample(50, random_state=42)
        logging.info("Background data for SHAP initialized.")

        # SHAP for Random Forest
        rf_explainer = shap.TreeExplainer(self.rf_model)
        rf_shap_values = rf_explainer.shap_values(X_train)
        logging.info("SHAP explainer for Random Forest initialized.")
        shap.summary_plot(rf_shap_values, X_train, plot_type="bar", show=False)
        plt.savefig(f'{self.save_path}/{self.identifier_rf}_shap_summary.png')
        plt.close()
        shap.summary_plot(rf_shap_values, X_train, show=False)
        plt.savefig(f'{self.save_path}/{self.identifier_rf}_shap_beeswarm.png')
        plt.close()

        # SHAP for Linear Regression
        linear_explainer = shap.LinearExplainer(self.linear_model, background_data)
        linear_shap_values = linear_explainer.shap_values(X_train)
        logging.info("SHAP explainer for Linear Regression initialized.")
        shap.summary_plot(linear_shap_values, X_train, plot_type="bar", show=False)
        plt.savefig(f'{self.save_path}/{self.identifier_linear}_shap_summary.png')
        plt.close()
        shap.summary_plot(linear_shap_values, X_train, show=False)
        plt.savefig(f'{self.save_path}/{self.identifier_linear}_shap_beeswarm.png')
        plt.close()

        logging.info("Finished feature importance evaluation.")
        return rf_top_features, linear_top_features

    def plot(self) -> None:
        """ Plot """
        logging.info("Starting plotting feature importances...")
        # Bar plot for Random Forest feature importances
        plt.figure(figsize=(10, 6))
        rf_features = list(self.rf_importances.keys())
        rf_importances = list(self.rf_importances.values())
        plt.barh(rf_features, rf_importances, color='b')
        plt.xlabel('Importance')
        plt.title('Random Forest Feature Importances')
        plt.gca().invert_yaxis()
        plt.savefig(f'{self.save_path}/{self.identifier_rf}_feature_importances.png')
        plt.close()

        # Bar plot for Linear Regression feature importances
        plt.figure(figsize=(10, 6))
        linear_features = list(self.linear_importances.keys())
        linear_importances = list(self.linear_importances.values())
        plt.barh(linear_features, linear_importances, color='r')
        plt.xlabel('Importance')
        plt.title('Linear Regression Feature Importances')
        plt.gca().invert_yaxis()
        plt.savefig(f'{self.save_path}/{self.identifier_linear}_feature_importances.png')
        plt.close()
        logging.info("Finished plotting feature importances.")

if __name__ == "__main__":
    logging.info("Starting main execution...")
    folder_path = "/home/georg/Documents/Master/Data_Literacy"
    data_df = pd.read_csv(folder_path + "/city_listings.csv")
    safe_path = folder_path + "/DataLit-InsideAirbnb" + "/results" + "/regressions"
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

    model = RegressionModels(
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
    logging.info("Finished main execution.")
    