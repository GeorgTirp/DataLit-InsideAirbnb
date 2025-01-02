import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

# Setting Hyperparameters and features
RandomForest_Hparams = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42
}

Feature_Selection = {
    'features': ['feature1', 'feature2'],
    'target': 'target'
}

test_split_size= 0.2

class RegressionModels:
    """ Compute evaluate and visualize regression models (current: Random Forest and Linear Regression)"""
    def __init__(
            self,
            data_df, 
            rf_hparams=RandomForest_Hparams, 
            Feature_Selection=Feature_Selection, 
            test_split_size=test_split_size):
        
        self.rf_hparams = rf_hparams
        self.linear_model = None
        self.rf_model = None
        X = data_df[Feature_Selection['features']]
        y = data_df[Feature_Selection['target']]
        self.train_split = train_test_split(X, y, test_size=test_split_size, random_state=42)


    def fit(self):
        """ Train and predict using Linear Regression and Random Forest"""
    
        X_train, X_test, y_train, y_test = self.train_split

        # Linear Regression
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        rf_model = RandomForestRegressor(**self.rf_hparams)
        rf_model.fit(X_train, y_train)

        self.linear_model = linear_model
        self.rf_model = rf_model
       




    def evaluate(self):
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
        
        return rf_mse, rf_r2, linear_mse, linear_r2, rf_cv_mean_score, linear_cv_mean_score
    


if __name__ == "__main__":
   pass