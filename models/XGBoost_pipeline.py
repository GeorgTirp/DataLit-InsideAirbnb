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
from typing import Tuple, Dict
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import re
#from add_custom_features import AddCustomFeatures
import logging
import os


# A function to remove outliers
def remove_outliers(X, y):
    return X, y

# Function to log and log_and_print messages
def log_and_print(message):
    print(message)
    logging.info(message)


# A function to remove correlated features
def remove_correlated_features(X, threshold):
    return X


def plot_results(results_df, r_score, p_value, save_results=False, save_path='results/', identifier=''):
    """
    Plots the results of the predictions.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing the results of the predictions.
    identifier : str
        Identifier for the plot.
    """

    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['y_test'], results_df['y_pred'], alpha=0.5)
    plt.plot([results_df['y_test'].min(), results_df['y_test'].max()], 
             [results_df['y_test'].min(), results_df['y_test'].max()], 
             color='red', linestyle='--', linewidth=2)
    plt.text(results_df['y_test'].min(), results_df['y_pred'].max(), f'R^2: {r_score:.2f}\nP-value: {p_value:.2e}', 
             fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.xlabel('Actual prices')
    plt.ylabel('Predicted prices')
    plt.title(f'Actual vs Predicted prices ({identifier})')
    plt.grid(True)
    plt.savefig(f'{save_path}/{identifier}_actual_vs_predicted.png')
    plt.show()




def run_XGBoost_pipeline(data='', target='listing_price', features=[], 
                     outlier_removal=False, cv=5, correlation_threshold=1, save_results=False, save_path='results/', add_custom_features=[], identifier='', random_state=42):
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
    - The SHAP prices
    - The Regresser performance
    """
    # Create results directory if it doesn't exist
    
    save_path = f'{save_path}/{identifier}' 
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Set up logging
    logging.basicConfig(filename=f'{save_path}/{identifier}_pipeline.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



    log_and_print('Starting the XGBoost pipeline...')
    log_and_print('----------------------------------')
    log_and_print('Parameters:')
    log_and_print(f'Data: {data}')
    log_and_print(f'Target: {target}')
    log_and_print(f'Features: {features}')
    log_and_print(f'Outlier removal: {outlier_removal}')
    log_and_print(f'CV: {cv}')
    log_and_print(f'Correlation threshold: {correlation_threshold}')
    log_and_print(f'Save results: {save_results}')
    log_and_print(f'Additional custom features: {add_custom_features}')
    log_and_print(f'Save path: {save_path}')
    log_and_print(f'Identifier: {identifier}')
    log_and_print(f'Random state: {random_state}')
    log_and_print('----------------------------------')

    ### -- Load and preprocess the data -- ###
    data = pd.read_csv(data)

    data['price'] = data['price'].replace('[\$,]', '', regex=True).astype(float)
    data = data[data['price'] < 1000]

    # Add custom features
    #Feature_Adder = AddCustomFeatures(data, add_custom_features)
    #data = Feature_Adder.return_data()


    # Extract the target variable
    y = data[target]


    # Extract the features
    if len(features) == 0:
        X = data.drop(columns=[target])
        log_and_print(f'Using all the features except {target}')
    else:
        log_and_print(f'Using the following features: {features}')
        log_and_print(data.columns)
        log_and_print(features + add_custom_features)
        X = data[features + add_custom_features]

    # Remove outliers
    if outlier_removal:
        X, y = remove_outliers(X, y)
        log_and_print('Outliers removed')

    # Remove correlated features
    X = remove_correlated_features(X, correlation_threshold)

    log_and_print('Features:')
    log_and_print(X.head())
    log_and_print('----------------------------------')
    log_and_print('Target:')
    log_and_print(y.head())
    log_and_print('----------------------------------')

    log_and_print(f'Number of samples: {len(X)}')
    log_and_print('----------------------------------')

    # Plot histograms of the data
    X.hist(bins=30, figsize=(20, 15))
    plt.suptitle('Feature Histograms', fontsize=16)
    plt.show()

    # Plot the target variable
    plt.hist(y, bins=30)
    plt.title('Target Variable Histogram')
    plt.show()



    # Safe the preprocessed data
    if save_results == True:
        X.to_csv(f'data/{identifier}_X.csv', index=False)
        log_and_print(f'Preprocessed data saved as data/{identifier}_X.csv')

    ### ---------------------------------- ###


    ### -- Train the model -- ###


    log_and_print('Training the model...')
    hyperparameter_cv = KFold(n_splits=cv, shuffle=True, random_state=42)
    model_evaluation_cv = KFold(n_splits=cv, shuffle=True, random_state=42)

    #XGBoost hyperparameters grid    
    param_grid_xgb = {
    #    'n_estimators': [150, 200, 500, 1000, 50],
        'learning_rate': [0.001, 0.01, 0.1, 0.2],
    #    'max_depth': [5, 6, 7, 8, 9, 10],
    #    'subsample': [0.6, 0.8, 1.0],
    #    'colsample_bytree': [0.6, 0.8, 1.0]
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    }

    # log_and_print the Hyperparamer grid
    log_and_print('----------------------------------')

    log_and_print("Hyperparameter grid:")
    for param, values in param_grid_xgb.items():
        log_and_print(f"{param}: {values}")
    log_and_print('----------------------------------')

    xgb = XGBRegressor(random_state=random_state)

    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, scoring='neg_root_mean_squared_error', cv=hyperparameter_cv, n_jobs=-1)

    all_y_test = []
    all_y_pred = []
    all_mse = []
    all_shap_prices = []
    fold = 1
    for train_idx, test_idx in model_evaluation_cv.split(X):
        log_and_print('----------------------------------')
        log_and_print(f'Training fold [{fold}/{cv}]:')
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)

        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)
        
        mse = mean_squared_error(y_test, y_pred)
        all_mse.append(mse)

        log_and_print(f'Mean Squared Error: {mse} for fold [{fold}/{cv}]')
        log_and_print(f'Best hyperparameters: {grid_search.best_params_}')
        log_and_print(f'Estimating SHAP values...')
        
        # Compute SHAP prices for the entire dataset using the best model
        explainer = shap.TreeExplainer(best_model)
        shap_prices = explainer.shap_values(X)
        all_shap_prices.append(shap_prices)
        log_and_print(f'SHAP values estimated for fold [{fold}/{cv}]')
        log_and_print('----------------------------------')
        fold += 1

    log_and_print('All folds trained')
    log_and_print('Evaluating Model...')

    ### --------------------- ###

    ### -- Evaluate the model -- ###

    # Calculate Pearson correlation and p-price
    r_score, p_price = pearsonr(all_y_test, all_y_pred)
    log_and_print(f'Pearson correlation: {r_score}, p-value: {p_price}')

    average_mse = np.mean(all_mse)
    log_and_print(f'Nested CV Mean Squared Error: {average_mse}')
        
    # Create a dataframe with the results for plotting
    results_df = pd.DataFrame({'y_test': all_y_test, 'y_pred': all_y_pred})

    if save_results == True:
        results_df.to_csv(f'{save_path}/{identifier}_results.csv', index=False)
        log_and_print(f'Results saved as results/{identifier}_results.csv')
        plot_results(results_df, r_score, p_price, save_results=True, save_path=save_path, identifier=identifier)

    # Aggregate SHAP prices across folds
    all_shap_prices = np.array(all_shap_prices)
    mean_shap_prices = np.mean(all_shap_prices, axis=0)
    if save_results:
        np.save(f'{save_path}/{identifier}_mean_shap_values.npy', mean_shap_prices)
        log_and_print(f'Mean SHAP values saved as {save_path}/{identifier}_mean_shap_values.npy')

    # Plot aggregated SHAP values (Feature impact)
    shap.summary_plot(mean_shap_prices, features=X, feature_names=X.columns, show=False)
    plt.title(f'{identifier} SHAP Summary Plot (Aggregated)', fontsize=16)
    if save_results:
        plt.subplots_adjust(top=0.90)
        plt.savefig(f'{save_path}/{identifier}_shap_aggregated_beeswarm.png')
        plt.close()


    # Plot aggregated SHAP values as bar plot (Feature importance)
    shap.summary_plot(mean_shap_prices, features=X, plot_type='bar', feature_names=X.columns, show=False)
    plt.title(f'{identifier} SHAP Summary Plot (Aggregated)', fontsize=16)
    if save_results:
        plt.subplots_adjust(top=0.90)
        plt.savefig(f'{save_path}/{identifier}_shap_aggregated_bar.png')
        plt.close()
    else:
        plt.show()









    