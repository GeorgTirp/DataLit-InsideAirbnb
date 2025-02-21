import pandas as pd
import sys
from typing import Tuple, Dict
from XGBoost_pipeline import run_XGBoost_pipeline


#######-- Set the parameters for the analysis --#######
# Preprocessed data
data = '/home/sn/pCloudDrive/AirBnB_Daten/Preprocessed_data/germany_preprocessed/berlin/city_listings_custom_features.csv'

# Name of the variable to predict in the data table
target = 'price'

# Add custom features, not provided by AirBnb? currently supported: ['distance_to_city_center', 'average_review_length']
add_custom_features = []

# Name of the variables to use for the prediction
# Load the data to determine numeric features
df = pd.read_csv(data, on_bad_lines="skip")
numeric_features = df.select_dtypes(include=['number']).columns.tolist()
features = [col for col in numeric_features if col != target]
print(features)

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
identifier = 'Berlin_with_custom_features_XGBoost'

# Random state
random_state = 42

#######----------------------------------------#######


### Run the pipeline with the specified paramters
run_XGBoost_pipeline(data=data, target=target, features=features, 
                     outlier_removal=outlier_removal, cv=cv, correlation_threshold=correlation_threshold, save_results=True, 
                     save_path=safe_path, identifier=identifier, add_custom_features=add_custom_features, random_state=random_state)
