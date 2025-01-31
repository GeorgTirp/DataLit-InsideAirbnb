
from tabpfn import TabPFNClassifier
from tabpfn import TabPFNRegressor
import numpy as np
import pandas as pd
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
from scipy.stats import pearsonr


class TabPFNRegression():
    """ Fit, evaluate, and get attributions regression models (current: Random Forest and Linear Regression)"""
    def __init__(
            self,
            data_df: pd.DataFrame, 
            Feature_Selection: dict, 
            test_split_size:float = 0.2,
            save_path: str = None,
            identifier: str = None):
        
        self.save_path = save_path
        self.identifier = identifier
        self.Feature_Selection = Feature_Selection

        self.reg_model = None
        self.X, self.y = self.model_specific_preprocess(data_df)
        self.train_split = train_test_split(self.X, self.y, test_size=test_split_size, random_state=42)
        self.metrics = None
        

    def model_specific_preprocess(self, data_df: pd.DataFrame, y: pd.DataFrame = None, Feature_Selection: dict = None) -> Tuple:
        """ Preprocess the data for the TabPFN model"""
        # Ensure all features are numeric
        if Feature_Selection is None:
            Feature_Selection = self.Feature_Selection
        data_df = data_df.dropna(subset=Feature_Selection['features'] + [Feature_Selection['target']])
        X = data_df[Feature_Selection['features']]
        y = data_df[Feature_Selection['target']]
        X = X.apply(pd.to_numeric, errors='coerce')
        # Remove dollar sign and convert to float
        if y.dtype == object:
            y = y.replace('[\$,]', '', regex=True).astype(float)
    
        return X, y
    
    def fit(self) -> None:
        """ Train and predict using Linear Regression and Random Forest"""
        reg = TabPFNRegressor()
        reg.fit(self.X, self.y)
        self.reg_model = reg
        return self.reg_model
       
    def predict(self, X_in: pd.DataFrame, save_results=False) -> Dict:
        """Predict using the trained model"""
        if self.reg_model is None:
            raise ValueError("Model not fitted yet")
        #X_in = X_in[Feature_Selection['features']]
        predictions = self.reg_model.predict(X_in)

        if save_results == True:
            # Optionally save predictions
            results_df = pd.DataFrame({'y_test': y, 'y_pred': predictions})
            results_df.to_csv(f'{self.save_path}/{self.identifier_rf}_results.csv', index=False)
        

        return predictions

    def evaluate(self) -> Tuple:
        """ Evaluate the models using mean squared error, r2 score and cross validation"""
        X_train, X_test, y_train, y_test = self.train_split

        pred = self.reg_model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        r2, p_price = pearsonr(y_test, pred)
        
        # Cross-validation for Random Forest
        #reg_cv_scores = cross_val_score(self.reg_model, X_train, y_train, cv=10, scoring='accuracy')
        #reg_cv_mean_score = reg_cv_scores.mean()

        print(f"TabPFN MSE: {mse}, R2: {r2}")
        print(f"TabPFN R^2: {r2}")
    
        tabpfn_metrics = {
            'mse': mse,
            'r2': r2,
            'p_value': p_price
        }
        self.metrics = tabpfn_metrics

        return tabpfn_metrics
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    def feature_importance(self, save_results=True) -> Dict:
        """Return the feature importance for the Random Forest and linear model"""
        logging.info("Starting feature importance evaluation.")

        X_train, X_test, y_train, y_test = self.train_split

        def loco_importances(X_train, y_test):
            logging.info("Starting Loco importance evaluation...")
            importances = {}
            for i, feature in enumerate(X_train.columns):
                logging.info(f"Evaluating Loco importance for feature {i + 1}/{len(X_train.columns)}: {feature}")

                # Remove the feature from the data
                X_train_loco = X_train.drop(columns=[feature])
                X_test_loco = X_test.drop(columns=[feature])

                # Train the model and get predictions
                self.reg_model.fit(X_train_loco, y_train)
                loco_pred = self.reg_model.predict(X_test_loco)

                # Compute the MSE and store the importance
                loco_mse = mean_squared_error(y_test, loco_pred)
                importances[feature] = abs(loco_mse - self.metrics['mse']) / self.metrics['mse']

                # Log progress for every 10th feature evaluated
                if (i + 1) % 10 == 0 or (i + 1) == len(X_train.columns):
                    logging.info(f"Progress: {i + 1}/{len(X_train.columns)} features evaluated.")

            logging.info("Finished Loco importance evaluation.")
            return importances

        def shap_importances():
            logging.info("Starting SHAP importance evaluation...")

            # Initialize SHAP
            shap.initjs()

            # Sample background data
            background_data = self.X.sample(50, random_state=42)
            logging.info("Background data for SHAP initialized.")

            # Initialize SHAP Explainer
            shap_values = []
            explainer = shap.DeepExplainer(self.model, background_data)
            for row in tqdm(self.X.itertuples(index=False), total=len(self.X), desc="Computing SHAP values"):
                row_shap = explainer.shap_values(pd.DataFrame([row], columns=self.X.columns))
                shap_values.append(row_shap)
            # Convert list of arrays to a single array
            shap_values = np.array(shap_values).squeeze()
            # Plot SHAP summary

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
            logging.info("SHAP summary plot generated.")
        
            return shap_values

        # Run Loco and SHAP importance evaluations
        logging.info("Evaluating SHAP feature importances...")
        shap_attributions = shap_importances()
        logging.info("SHAP importance evaluation completed.")
    
        logging.info("Evaluating LOCO feature importances...")
        loco_attributions = loco_importances()
        logging.info("LOCO importance evaluation completed.")
    
        # Save results if specified
        if save_results:
            logging.info(f"Saving results to {self.save_path}/{self.identifier}_mean_shap_values.npy")
            np.save(f'{self.save_path}/{self.identifier}_mean_shap_values.npy', shap_attributions)
    
        return loco_attributions, shap_attributions

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

if __name__ == "__main__":
    #folder_path = "/Users/georgtirpitz/Documents/Data_Literacy"
    folder_path = "/home/georg/Documents/Master/Data_Literacy"
    data_df = pd.read_csv(folder_path + "/city_listings.csv")
    identifier = "test_TabPFN"
    safe_path = folder_path + "/DataLit-InsideAirbnb" + "/results/" + identifier + "/"
    
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
    preds = model.predict(X)
    metrics = model.evaluate()
    importances = model.feature_importance()
    