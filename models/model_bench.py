from FT_Transformer import FT_Transfomer
from base_regressions import LinearRegressionModel
from base_regressions import RandomForestModel
from TabPFN import TabPFNRegression
import pandas as pd
import numpy as np
import os
import logging
import gc
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR

# Define the hyperparameter sets for FT Transformer
ft_transformer_hparams = [
    {
        "n_blocks": 2,
        "d_block": 200,
        "attention_n_heads": 5,
        "attention_dropout": 0.2,
        "ffn_d_hidden": None,
        "ffn_d_hidden_multiplier": 4 / 3,
        "ffn_dropout": 0.3,
        "residual_dropout": 0.0
    },
    {
        "n_blocks": 2,
        "d_block": 400,
        "attention_n_heads": 8,
        "attention_dropout": 0.2,
        "ffn_d_hidden": None,
        "ffn_d_hidden_multiplier": 2 / 3,
        "ffn_dropout": 0.2,
        "residual_dropout": 0.0
    },
    {
        "n_blocks": 4,
        "d_block": 250,
        "attention_n_heads": 10,
        "attention_dropout": 0.1,
        "ffn_d_hidden": None,
        "ffn_d_hidden_multiplier": 1 / 3,
        "ffn_dropout": 0.2,
        "residual_dropout": 0.1
    },
    {
        "n_blocks": 7,
        "d_block": 300,
        "attention_n_heads": 10,
        "attention_dropout": 0.1,
        "ffn_d_hidden": None,
        "ffn_d_hidden_multiplier": 4 / 3,
        "ffn_dropout": 0.1,
        "residual_dropout": 0.1
    }
]
max_depths = [10, 20, 30, 40]
batch_sizes = [5, 10, 256 ,2048]
n_epochs_list = [40 ,40, 40, 50]

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
logging.disable(logging.CRITICAL)  # Disable all logging calls
logging.info("Script started")
folder_path = "/kaggle"
data_df = pd.read_csv(folder_path + "/input/datalit-dataset/european_cities_data_-40000_london.csv")
identifier = "model_comparison"
safe_path = folder_path + "/working/" + identifier + "/"
Feature_Selection = {
    'features': [
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

        ],
        'target': 'price'
    }
feature_types = {
    "continuous": Feature_Selection["features"],
    "categorical": []
}


n_top_features = 15
# Define the models and their names
keys = ["size", "Linear Regression", "Random Forest", "TabPFN", "FT-Transformer"]
results = {key: [] for key in keys}

test_split_size = 0.2
n = len(data_df)
num_eval_steps = 12
num_hparams = 4
steps_per_hparam = num_eval_steps // num_hparams
eval_step = n // num_eval_steps
start_value = 100
indices = np.logspace(np.log10(start_value), np.log10(n), num=num_eval_steps).astype(int)
seeds = np.linspace(1, 12)
print(indices)
# Outer loop with progress bar for evaluation steps
for i in tqdm(range(num_eval_steps), desc="Evaluation Steps", unit="step"):
    current_size = indices[i]
    print(f'Current data size: {current_size}')

    index = min(i // steps_per_hparam, num_hparams - 1)
    # Randomly sample the data without replacement
    seed = int(seeds[i])
    data_df = data_df.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle once
    data = data_df.sample(n=current_size, replace=False)  # Use cumulative data up to the current step
    tab_size = min(current_size, 10000)
    data_tab = data_df.sample(n=tab_size, replace=False)
    
    results["size"].append(len(data))
    # Linear Regression Model
    linear_model = LinearRegressionModel(
        data, 
        Feature_Selection, 
        test_split_size, 
        safe_path, 
        identifier, 
        n_top_features)
    linear_model.fit()
    linear_model.evaluate()
    results["Linear Regression"].append(linear_model.metrics["r2"])
    del linear_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Random Forest Model
    max_depth = max_depths[index]
    RandomForest_Hparams = {
    'n_estimators': 100,
    'max_depth': max_depth,
    'random_state': 42
    }
    rf_model = RandomForestModel(
        data, 
        Feature_Selection, 
        RandomForest_Hparams, 
        test_split_size, 
        safe_path, 
        identifier, 
        n_top_features)
    rf_model.fit()
    rf_model.evaluate()
    results["Random Forest"].append(rf_model.metrics["r2"])
    del rf_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # TabPFN Model
    tab_model = TabPFNRegression(data_tab, Feature_Selection, test_split_size, safe_path, identifier)
    tab_model.fit()
    tab_model.evaluate()
    results["TabPFN"].append(tab_model.metrics["r2"])
    del tab_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # FT Transformer Model
    # Use a different set of hyperparameters for each evaluation step
    
    
    index = min(i // steps_per_hparam, num_hparams - 1)
    print(index)
    # Ensure it stays within bounds
    hparams = ft_transformer_hparams[index]
    print(f'Current FTT hparams: {hparams}')
    
    batch_size = batch_sizes[index]
    print(f'Current batch size: {batch_size}')
    
    n_epochs =  n_epochs_list[index]
    print(f'Current epochs: {n_epochs}')
    
    ft_model = FT_Transfomer(
        data, 
        Feature_Selection,
        test_split_size,
        safe_path,
        identifier,
        hparams,
        feature_types)
    
    # Set up optimizer and scheduler for FT Transformer
    param_groups = [
        {
            "params": [p for n, p in ft_model.model.named_parameters() if "bias" not in n and "LayerNorm" not in n],
            "weight_decay": 1e-3,
        },
        {
            "params": [p for n, p in ft_model.model.named_parameters() if "bias" in n or "LayerNorm" in n],
            "weight_decay": 0.0,
        },
    ]
    data_processed, n_cont_features, cat_cardinalities = ft_model.model_specific_preprocess(data, Feature_Selection)
    dataset_size = len(data_processed["train"]["y"])
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=0.001,
        betas=(0.9, 0.99),
        eps=1e-08,
        amsgrad=True
    )
    second_order_method = False
    n_epochs = n_epochs
    batch_size = batch_size
    steps_per_epoch = dataset_size // batch_size
    total_steps = steps_per_epoch * n_epochs
    warmup_steps = int(0.05 * total_steps)
    cosine_steps = max(1, total_steps - warmup_steps)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    # Train and evaluate FT Transformer
    ft_model.train(
        batch_size=batch_size, 
        patience=150, 
        n_epochs=n_epochs, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        second_order_method=second_order_method
    )
    ft_model.predict(data, evaluate=True)
    results["FT-Transformer"].append(ft_model.metrics["r2"])
    del ft_model
    del optimizer
    del scheduler
    del param_groups
    gc.collect()
    torch.cuda.empty_cache()

    # Save results to CSV
    pd.DataFrame(results).to_csv(f'{safe_path}/{identifier}.csv')

def plot_r2_scores(results):
    """
    Plots the R² scores for each model over evaluation steps.

    Parameters:
    results (dict): A dictionary where keys are model names and values are lists of R² scores.
    """
    folder_path = "/kaggle"
    identifier = "model_comparison"
    safe_path = folder_path + "/working/" + identifier + "/"
    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Plot each model's R² scores
    for model_name, r2_scores in results.items():
        plt.plot(r2_scores, label=model_name, marker='o')

    # Add labels, title, and legend
    plt.xlabel("Evaluation Step")
    plt.ylabel("R² Score")
    plt.title("R² Scores Over Evaluation Steps")
    plt.legend()
    plt.savefig(f'{safe_path}/{identifier}.png')
    # Show the plot
    plt.grid(True)
    plt.show()
