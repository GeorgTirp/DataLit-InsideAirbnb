import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the custom color palette from your image
custom_palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#525252"]

# Set Seaborn style, context, and custom palette
sns.set_theme(style="whitegrid", context="paper")
sns.set_palette(custom_palette)

path = '/Users/georgtirpitz/Documents/Data_Literacy/DataLit-InsideAirbnb/results/Model_benchmark/model_comparison_fixed_r.csv'

# Read in the CSV
results = pd.read_csv(path)
# Drop the 'Unnamed' and 'size' columns if they exist
if 'Unnamed: 0' in results.columns:
    results = results.drop(columns=['Unnamed: 0'])
if 'size' in results.columns:
    results = results.drop(columns=['size'])
# Create a figure

plt.figure(figsize=(6, 4))
sample_sizes = [100, 183, 337, 621, 1142, 2099, 3860, 7097, 13047, 23986, 44095, 81066]

# Create a figure
plt.figure(figsize=(6, 4))

# Plot each model's R² scores in a loop, using sample_sizes on the x-axis
for model_name, r2_scores in results.items():
    sns.lineplot(x=sample_sizes, y=r2_scores, label=model_name, marker='o')

# Optionally use a log scale for the x-axis if you want to emphasize the “logarithmic” nature
plt.xscale("log")
plt.xticks(sample_sizes, sample_sizes)
# Label the axes and set the title
plt.xlabel("Sample Size")
plt.ylabel("R² Score")
plt.title("R² Scores Over Evaluation Steps")
plt.legend()
plt.tight_layout()

# Save and show the plot
plt.savefig('model_comparison.png', dpi=300)
plt.show()


