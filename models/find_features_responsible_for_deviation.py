import numpy as np
import pandas as pd
import joblib
import shap

# Load data
train_data = pd.read_csv('/media/sn/Frieder_Data/Master_Machine_Learning/data_preprocessed/european_cities_data_-40000_london.csv')
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
    "oslo"
]
train_data = train_data[features]

shap_train_data = pd.read_csv('results/XGBoost_all_european_cities_no_cv/XGBoost_all_european_cities_no_cv_shap_values.csv')

inference_data_original = pd.read_csv('/media/sn/Frieder_Data/Master_Machine_Learning/data_preprocessed/london_inference_data.csv')
inference_data = inference_data_original[features]  # Avoid missing values

# Define the listing to analyze
listing = 0

# Load the model
model = joblib.load('results/XGBoost_all_european_cities_no_cv/XGBoost_all_european_cities_no_cv_best_model.joblib')

# Perform inference
predictions = model.predict(inference_data.iloc[[listing]])

# Create a SHAP explainer
explainer = shap.TreeExplainer(model, train_data)

# Get SHAP values for the prediction
shap_values = explainer(inference_data.iloc[[listing]])

# Print and plot
import matplotlib.pyplot as plt
# Adjust the plot size
plt.figure(figsize=(12, 6))

# Plot SHAP waterfall plot
shap.plots.waterfall(shap_values[0], show=False)

# Add prediction and actual price to the plot
predicted_price = predictions[0]
actual_price = inference_data_original.iloc[listing]['price'] if 'price' in inference_data_original.columns else 'N/A'
plt.title(f'Predicted Price: {predicted_price}, Actual Price: {actual_price}')

# Adjust layout to prevent cutting off labels
plt.tight_layout()

plt.show()