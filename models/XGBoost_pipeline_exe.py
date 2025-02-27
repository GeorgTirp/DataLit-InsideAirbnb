import pandas as pd
import sys
from typing import Tuple, Dict
from XGBoost_pipeline import run_XGBoost_pipeline
from XGBoost_pipeline_no_cv import run_XGBoost_pipeline_no_cv


#######-- Set the parameters for the analysis --#######
# Preprocessed data
data = '/media/sn/Frieder_Data/Master_Machine_Learning/data_preprocessed/european_cities_data_-40000_london.csv'

# Name of the variable to predict in the data table
target = 'price'

# Add custom features, not provided by AirBnb? currently supported: ['distance_to_city_center', 'average_review_length']
add_custom_features = []

# Name of the variables to use for the prediction
# Load the data to determine numeric features
df = pd.read_csv(data, on_bad_lines="skip")
features = [
    "host_response_rate",
    "host_acceptance_rate",
    "host_listings_count",
    "host_total_listings_count",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "minimum_nights",
    "maximum_nights",
    "minimum_minimum_nights",
    "maximum_minimum_nights",
    "minimum_maximum_nights",
    "maximum_maximum_nights",
    "minimum_nights_avg_ntm",
    "maximum_nights_avg_ntm",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "number_of_reviews_l30d",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms",
    "reviews_per_month",
    "distance_to_city_center",
    "average_review_length",
    "spelling_errors",
    "host_profile_pic_people_visible",
    "host_profile_pic_male_or_female",
    "host_profile_pic_setting_indoor_outdoor",
    "host_profile_pic_professionality",
    "host_profile_pic_quality",
    "aesthetic_score",
    "picture_url_setting_indoor_outdoor",
    "amount_of_amenities",
    "berlin",
    "barcelona",
    "istanbul",
    "london",
    "oslo",
    "host_is_superhost_t",
    "host_has_profile_pic_t",
    "host_identity_verified_t",
    "has_availability_t",
    "instant_bookable_t",
    "host_response_time_a few days or more",
    "host_response_time_not available",
    "host_response_time_within a day",
    "host_response_time_within a few hours",
    "host_response_time_within an hour",
    "room_type_Entire home/apt",
    "room_type_Hotel room",
    "room_type_Private room",
    "room_type_Shared room"
]

# Outlier removal?
outlier_removal = False

# Number of cross-validation folds
cv = 5

# Correlation threshold for correlated features
correlation_threshold = 1

# Save the results?
save_results = True

# Safe path
safe_path = 'results/'

# Identifier
identifier = 'XGBoost_all_european_cities_no_cv'

# Random state
random_state = 42

#######----------------------------------------#######


### Run the pipeline with the specified paramters
run_XGBoost_pipeline_no_cv(data=data, target=target, features=features, 
                     outlier_removal=outlier_removal, cv=cv, correlation_threshold=correlation_threshold, save_results=True, 
                     save_path=safe_path, identifier=identifier, add_custom_features=add_custom_features, random_state=random_state)
