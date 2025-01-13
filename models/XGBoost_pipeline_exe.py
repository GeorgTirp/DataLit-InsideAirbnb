import pandas as pd
import sys
sys.path.append('preprocessing/')
from preprocessing import read
from typing import Tuple, Dict
from XGBoost_pipeline import run_XGBoost_pipeline


#######-- Set the parameters for the analysis --#######
# Preprocessed data
data = 'data/example_data'

# Name of the variable to predict in the data table
target = 'home_price'

# Name of the variables to use for the prediction
features = ['longitude', 'latitue']  # Emtpy list means all the variables except the target

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
                     safe_path=safe_path, identifier=identifier, random_state=random_state)
