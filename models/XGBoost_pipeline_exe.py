import pandas as pd
import sys
from typing import Tuple, Dict
from XGBoost_pipeline import run_XGBoost_pipeline


#######-- Set the parameters for the analysis --#######
# Preprocessed data
data = '/home/frieder/pCloudDrive/AirBnB_Daten/Preprocessed_data/germany_preprocessed/berlin/city_listings.csv'

# Name of the variable to predict in the data table
target = 'price'

# Name of the variables to use for the prediction
features = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'number_of_reviews', 'review_scores_value']  # Emtpy list means all the variables except the target

# Outlier removal?
outlier_removal = False

# Number of cross-validation folds
cv = 5

# Correlation threshold for correlated features
correlation_threshold = 0.9

# Save the results?
save_results = True

# Safe path
safe_path = 'results/'

# Identifier
identifier = 'test_run'

# Random state
random_state = 42

#######----------------------------------------#######


### Run the pipeline with the specified paramters
run_XGBoost_pipeline(data=data, target=target, features=features, 
                     outlier_removal=outlier_removal, cv=cv, correlation_threshold=correlation_threshold, save_results=True, 
                     save_path=safe_path, identifier=identifier, random_state=random_state)
