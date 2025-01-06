import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

from preprocessing.preprocessing_test import preprocess_data
from typing import Tuple, Dict




# Setting Hyperparameters and features
RandomForest_Hparams = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42
}

# Setting features and target
Feature_Selection = {
    'features': [
        'latitude', 
        'longitude'],

    'target': 'price'
}

# Setting test split size
test_split_size= 0.2

class RegressionModels:
    """ Fit, evaluate, and get attributions regression models (current: Random Forest and Linear Regression)"""
    def __init__(
            self,
            data_df: pd.DataFrame, 
            rf_hparams: dict = RandomForest_Hparams, 
            Feature_Selection: dict= Feature_Selection, 
            test_split_size:float = test_split_size):
        
        self.rf_hparams = rf_hparams
        self.linear_model = None
        self.rf_model = None
        X = data_df[Feature_Selection['features']]
        y = data_df[Feature_Selection['target']]
        self.train_split = train_test_split(X, y, test_size=test_split_size, random_state=42)


    def fit(self) -> None:
        """ Train and predict using Linear Regression and Random Forest"""
    
        X_train, X_test, y_train, y_test = self.train_split

        # Linear Regression
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        rf_model = RandomForestRegressor(**self.rf_hparams)
        rf_model.fit(X_train, y_train)

        self.linear_model = linear_model
        self.rf_model = rf_model
       
    def predict(self, X: pd.DataFrame) -> Tuple:
        """ Predict using the trained models (for class extern usage)"""
        rf_pred = self.rf_model.predict(X)
        linear_pred = self.linear_model.predict(X)
        return rf_pred, linear_pred

    def evaluate(self) -> Tuple:
        """ Evaluate the models using mean squared error, r2 score and cross validation"""
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

        print(f"Random Forest MSE: {rf_mse}, R2: {rf_r2}")
        print(f"Linear Regression MSE: {linear_mse}, R2: {linear_r2}")
        print(f"Random Forest 10-fold CV Mean R2 Score: {rf_cv_mean_score}")
        print(f"Linear Regression 10-fold CV Mean R2 Score: {linear_cv_mean_score}")

        linear_metrics = {
            'mse': linear_mse,
            'r2': linear_r2,
            'cv_mean_score': linear_cv_mean_score
        }

        rf_metrics = {
            'mse': rf_mse,
            'r2': rf_r2,
            'cv_mean_score': rf_cv_mean_score
        }

        return linear_metrics, rf_metrics
    
    def feature_importance(self, top_n: int = 10) -> Tuple[Dict, Dict]:
        """ Return the feature importance for the Random Forest and linear model"""
        # Get attributions
        rf_attribution = self.rf_model.feature_importances_
        linear_attribution = abs(self.linear_model.coef_) / sum(abs(self.linear_model.coef_))
        feature_names = Feature_Selection['features']
        rf_indices = rf_attribution.argsort()[-top_n:][::-1]
        linear_indices = linear_attribution.argsort()[-top_n:][::-1]

        # Get top n features
        rf_top_features = {feature_names[i]: rf_attribution[i] for i in rf_indices}
        linear_top_features = {feature_names[i]: linear_attribution[i] for i in linear_indices}
        return rf_top_features, linear_top_features

    def plot():
        """ Plot """
        pass

if __name__ == "__main__":
    folder_path = '/Users/georgtirpitz/Documents/Data_Literacy/example_data'
    data_df = preprocess_data(folder_path)
    model = RegressionModels(data_df)
    model.fit()
    model.evaluate()
    model.feature_importance(10)