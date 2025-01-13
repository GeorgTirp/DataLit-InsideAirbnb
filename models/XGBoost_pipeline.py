import pandas as pd
import shap
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut, train_test_split, KFold, cross_val_score
import sys
sys.path.append('preprocessing/')
from preprocessing import read
from typing import Tuple, Dict
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import numpy as np


# A function to remove outliers
def remove_outliers(X, y):
    return X, y


# A function to remove correlated features
def remove_correlated_features(X, threshold):
    return X




def run_XGBoost_pipeline(data='', target='listing_price', features=[], 
                     outlier_removal=False, cv=5, correlation_threshold=1, save_results=False, safe_path='results/', identifier='', random_state=42):
    """
    Runs a pipeline to predicts the target variable using an XGBoost regressor. The features are subsequently evaluated using SHAP analysis.

    Parameters
    ----------
    data : str
        Path to the data file.
    target : str
        Name of the variable to predict in the data table.
    features : list
        Name of the variables to use for the prediction.
    outlier_removal : bool
        If True, removes the outliers from the data.
    cv : int
        Number of cross-validation folds.
    correlation_threshold : float
        Correlation threshold for correlated features.
    safe_results : bool
        If True, saves the results.

    Returns
    -------
    - The result of the predictions
    - The feature importances
    - The SHAP values
    - The Regresser performance
    """

    ### -- Load and preprocess the data -- ###
    data = read(data)



    # Extract the target variable
    y = data[target]

    # Extract the features
    if len(features) == 0:
        X = data.drop(columns=[target])
    else:
        X = data[features]

    # Remove outliers
    if outlier_removal:
        X, y = remove_outliers(X, y)

    # Remove correlated features
    X = remove_correlated_features(X, correlation_threshold)


    # Safe the preprocessed data
    if save_results == True:
        X.to_csv(f'data/{identifier}_X.csv', index=False)

    ### ---------------------------------- ###


    ### -- Train the model -- ###
    
    hyperparameter_cv = KFold(n_splits=cv, shuffle=True, random_state=42)
    model_evaluation_cv = KFold(n_splits=cv, shuffle=True, random_state=42)

    #XGBoost hyperparameters grid    
    param_grid_xgb = {
        'n_estimators': [100, 150, 200],
        'learning_rate': [0.001, 0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    xgb = XGBRegressor(random_state=random_state)

    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, scoring='mean_squared_error', cv=hyperparameter_cv, n_jobs=-1)

    all_y_test = []
    all_y_pred = []
    all_mse = []
    all_shap_values = []
    for train_idx, test_idx in model_evaluation_cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)

        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)
        
        mse = mean_squared_error(y_test, y_pred)
        all_mse.append(mse)
        
        # Compute SHAP values for the entire dataset using the best model
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X)
        all_shap_values.append(shap_values)

    ### --------------------- ###

    ### -- Evaluate the model -- ###

    # Calculate Pearson correlation and p-value
    r_score, p_value = pearsonr(all_y_test, all_y_pred)
    print(f'Pearson correlation: {r_score}, p-value: {p_value}')

    average_mse = np.mean(all_mse)
    print(f'Nested CV Mean Squared Error: {average_mse}')
        
    # Create a dataframe with the results for plotting
    results_df = pd.DataFrame({'y_test': all_y_test, 'y_pred': all_y_pred})

    # Aggregate SHAP values across folds
    all_shap_values = np.array(all_shap_values)
    mean_shap_values = np.mean(all_shap_values, axis=0)
    mean_abs_shap_values = np.mean(np.abs(mean_shap_values), axis=0)









    