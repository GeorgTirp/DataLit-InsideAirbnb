
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


# Setting features and target
Feature_Selection = {
    'features': [
        'latitude', 
        'longitude'],

    'target': 'price'
}

# Setting test split size
test_split_size= 0.2

class TabPFNRegression():
    """ Fit, evaluate, and get attributions regression models (current: Random Forest and Linear Regression)"""
    def __init__(
            self,
            data_df: pd.DataFrame, 
            Feature_Selection: dict= Feature_Selection, 
            test_split_size:float = test_split_size):
        
        self.reg_model = None
        X = data_df[Feature_Selection['features']]
        y = data_df[Feature_Selection['target']]
        self.train_split = train_test_split(X, y, test_size=test_split_size, random_state=42)
        self.metrics = None

    def fit(self) -> None:
        """ Train and predict using Linear Regression and Random Forest"""
    
        X_train, X_test, y_train, y_test = self.train_split
        reg = TabPFNRegressor()
        reg.fit(X_train, y_train)

        self.reg_model = reg
        return self.reg_model
       
    def predict(self, X: pd.DataFrame) -> Dict:
        """ Predict using the trained models (for class extern usage)"""
        if self.reg_model is None:
            raise ValueError("Model not fitted yet")
        
        # Fit the Inference distribution
        X_train, X_test, y_train, y_test = self.train_split
        predictions = self.reg_model.predict(X_test)
        # Get the point estimate
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
    
    def feature_importance(self, top_n: int = 10) -> Dict:
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
        loco_attributions = loco_importances(self, X_train, y_test)

        return loco_attributions

    def plot():
        """ Plot """
        pass

if __name__ == "__main__":
    folder_path = "/home/georg/Documents/Master/Data_Literacy/DataLit-InsideAirbnb/"
    data_df = pd.read_csv(folder_path + "/data/germany_listings.csv")
    model = TabPFNRegression(data_df)
    model.fit()
    preds = model.predict(data_df)
    metrics = model.evaluate()
    importances = model.feature_importance()
    print(preds.shape)