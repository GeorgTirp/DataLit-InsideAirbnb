
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



class TabPFNRegression():
    """ Fit, evaluate, and get attributions regression models (current: Random Forest and Linear Regression)"""
    def __init__(
            self,
            data_df: pd.DataFrame, 
            Feature_Selection: dict= Feature_Selection, 
            test_split_size:float = test_split_size,
            save_path: str = None,
            identifier: str = None):
        
        self.reg_model = None
        X,y = self.model_specific_preprocess(data_df)
        self.train_split = train_test_split(X, y, test_size=test_split_size, random_state=42)
        self.metrics = None
        self.save_path = save_path
        self.identifier = identifier

    def model_specific_preprocess(self, data_df: pd.DataFrame) -> Tuple:
        """ Preprocess the data for the TabPFN model"""
        # Ensure all features are numeric
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
    
        X_train, X_test, y_train, y_test = self.train_split
        reg = TabPFNRegressor()
        reg.fit(X_train, y_train)
        self.reg_model = reg
        return self.reg_model
       
    def predict(self, X: pd.DataFrame, save_results=False) -> Dict:
        """ Predict using the trained models (for class extern usage)"""
        if self.reg_model is None:
            raise ValueError("Model not fitted yet")
        
        # Fit the Inference distribution
        X_train, X_test, y_train, y_test = self.train_split
        predictions = self.reg_model.predict(X_test)
        # Get the point estimate
        results_df = pd.DataFrame({'y_test': y_test, 'y_pred': predictions})
        if save_results == True:
            results_df.to_csv(f'{self.save_path}/{self.identifier}_results.csv', index=False)


        return predictions

    def evaluate(self) -> Tuple:
        """ Evaluate the models using mean squared error, r2 score and cross validation"""
        X_train, X_test, y_train, y_test = self.train_split

        # Linear Regression
        reg_pred = self.reg_model.predict(X_test)
        reg_mse = mean_squared_error(y_test, reg_pred)
        reg_r2 = r2_score(y_test, reg_pred)
        
        
        # Cross-validation for Random Forest
        #reg_cv_scores = cross_val_score(self.reg_model, X_train, y_train, cv=10, scoring='accuracy')
        #reg_cv_mean_score = reg_cv_scores.mean()

        
        print(f"TabPFN MSE: {reg_mse}, R2: {reg_r2}")
        print(f"TabPFN R^2: {reg_r2}")
        
    
        tabpfn_metrics = {
            'mse': reg_mse,
            'r2': reg_r2,
        }
        self.metrics = tabpfn_metrics

        return tabpfn_metrics
    
    def feature_importance(self, save_results=True) -> Dict:
        """ Return the feature importance for the Random Forest and linear model"""
        X_train, X_test, y_train, y_test = self.train_split
        def loco_importances(self, X_train, y_test):
            importances = {}
            for feature in X_train.columns:
                X_train_loco = X_train.drop(columns=[feature])
                X_test_loco = X_test.drop(columns=[feature])
                
                self.reg_model.fit(X_train_loco, y_train)
                loco_pred = self.reg_model.predict(X_test_loco)
                loco_mse = mean_squared_error(y_test, loco_pred)
                
                importances[feature] = abs(loco_mse - self.metrics['mse']) / self.metrics['mse']
            
            return importances

        def shap_importances(self, X_train, y_test):
            
            shap.initjs()
            background_data = X_train.sample(100, random_state=42)
            explainer = shap.KernelExplainer(self.reg_model, background_data)
            shap_values = explainer(X_train)
            shap.summary_plot(shap_values, X_train)
            return shap_values
        
        shap_attributions = shap_importances(self, X_train, y_test)
        loco_attributions = loco_importances(self, X_train, y_test)

    
        if save_results:
            np.save(f'{self.save_path}/{self.identifier}_shap_values.npy', shap_attributions)
            #log_and_print(f'Mean SHAP values saved as {self.save_path}/{self.identifier}_mean_shap_values.npy')
        
        if save_results:
            np.save(f'{self.save_path}/{self.identifier}_shap_values.npy', loco_attributions)
            #log_and_print(f'Mean SHAP values saved as {self.save_path}/{self.identifier}_mean_shap_values.npy')

        # Plot aggregated SHAP values (Feature impact)
        plt.title(f'{identifier} SHAP Summary Plot (Aggregated)', fontsize=16)
        if save_results:
            plt.subplots_adjust(top=0.90)
            plt.savefig(f'{self.save_path}/{self.identifier}_shap_aggregated_beeswarm.png')
            plt.close()


        # Plot aggregated SHAP values as bar plot (Feature importance)
        shap.summary_plot(shap_attributions, X_train, plot_type='bar', feature_names=X_train.columns, show=False, max_display=40)
        plt.title(f'{identifier} SHAP Summary Plot (Aggregated)', fontsize=16)
        if save_results:
            plt.subplots_adjust(top=0.90)
            plt.savefig(f'{self.save_path}/{identifier}_shap_aggregated_bar.png')
            plt.close()
        else:
            plt.show()
        return loco_attributions, shap_attributions

    def plot():
        """ Plot """
        pass

if __name__ == "__main__":
    folder_path = "/Users/georgtirpitz/Documents/Data_Literacy"
    data_df = pd.read_csv(folder_path + "/city_listings.csv")
    safe_path = folder_path + "/results"
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
    Feature_Adder = AddCustomFeatures(data_df, add_custom_features, Feature_Selection, test_split_size, safe_path, identifier)
    data_df = Feature_Adder.return_data()
    model = TabPFNRegression(data_df)
    model.fit()
    preds = model.predict(data_df)
    metrics = model.evaluate()
    importances = model.feature_importance()
    