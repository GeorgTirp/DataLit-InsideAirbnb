import pandas as pd
import joblib
from xgboost import XGBRegressor
import shap


# Define file paths
model_path = 'results/XGBoost_all_european_cities_no_cv/XGBoost_all_european_cities_no_cv_best_model.joblib'  
data_path = '/media/sn/Frieder_Data/Master_Machine_Learning/data_preprocessed/london_inference_data.csv'  

# Load the trained model
best_model = joblib.load(model_path)
print('Model loaded successfully.')

# Load the dataset for inference
input_data_original = pd.read_csv(data_path)
print('Input data loaded successfully.')


input_data = input_data_original[[
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
    "oslo"
]]


# Perform inference
predictions = best_model.predict(input_data)

# Save predictions to CSV
# Add the original price column back to the predictions DataFrame
if 'price' in input_data_original.columns:
    predictions_df = pd.DataFrame({
        'price': input_data_original['price'],
        'predictions': predictions,
        'latitude': input_data_original['latitude'],
        'longitude': input_data_original['longitude']
    })
else:
    predictions_df = pd.DataFrame({'predictions': predictions})
predictions_csv_path = 'results/XGBoost_all_european_cities_no_cv/london_inference_data.csv'  # Update with actual path
predictions_df.to_csv(predictions_csv_path, index=False)
print(f'Predictions saved at {predictions_csv_path}')

# Perform SHAP analysis
#explainer = shap.TreeExplainer(best_model)
#shap_values = explainer.shap_values(input_data)
#print('SHAP values calculated.')
#
## Save SHAP values to CSV
#shap_values_df = pd.DataFrame(shap_values, columns=input_data.columns)
#shap_values_csv_path = 'results/XGBoost_all_european_cities_no_cv/london_shap_values.csv'  # Update with actual path
#shap_values_df.to_csv(shap_values_csv_path, index=False)
#print(f'SHAP values saved at {shap_values_csv_path}')