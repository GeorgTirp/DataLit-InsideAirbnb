# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'InsideAirBnB'
copyright = '2025, Georg Tirpitz, Nils Klute, Matthis Nommensen, Frieder Wizgall'
author = 'Georg Tirpitz, Nils Klute, Matthis Nommensen, Frieder Wizgall'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
# Import the RSTGenerator class
import os

class RSTGenerator:
    """
    Class to generate a single .rst documentation file for all folders in the results directory.
    """
    def __init__(self, results_dir, output_file):
        """
        Initialize the RSTGenerator.

        Args:
            results_dir (str): Path to the results directory.
            output_file (str): Path to the output .rst file.
        """
        self.results_dir = results_dir
        self.output_file = output_file

    def generate_rst(self):
        """
        Generate the .rst documentation.
        """
        header = "Results\n=======\n\n"
        sections = []

        # Iterate over folders in the results directory
        for folder in sorted(os.listdir(self.results_dir)):
            folder_path = os.path.join(self.results_dir, folder)
            if os.path.isdir(folder_path):
                # Header for the section
                sections.append(f"XGBoost Regressor {folder.capitalize()}\n{'-' * (len('XGBoost Regressor ') + len(folder.capitalize()))}\n\n")

                # Log file inclusion
                log_file = os.path.join(folder_path, f"{folder}_pipeline.log")
                if os.path.exists(log_file):
                    sections.append(f"The parameters of the model:\n\n")
                    sections.append(f".. literalinclude:: ../results/{folder}/{folder}_pipeline.log")
                    sections.append(f"   :caption: Log")
                    sections.append(f"   :lines: 1-15\n\n")

                # Figures inclusion
                figures = [
                    ("actual_vs_predicted", "The actual vs predicted values of the model. With the Pearson correlation coefficient and p-value."),
                    ("shap_aggregated_beeswarm", "The SHAP values of the model. The effects of the individual features can be read from this plot."),
                    ("shap_aggregated_bar", "The absolute SHAP values of the model. The feature importances can be read from this plot.")
                ]

                for fig, caption in figures:
                    fig_path = os.path.join(folder_path, f"{folder}_{fig}.png")
                    if os.path.exists(fig_path):
                        sections.append(f".. figure:: ../results/{folder}/{folder}_{fig}.png")
                        sections.append(f"   :alt: {fig.replace('_', ' ').title()}\n")
                        sections.append(f"   {caption}\n\n")

        # Combine everything into the final .rst file
        with open(self.output_file, "w") as f:
            f.write(header + "\n".join(sections))


results_dir = "../results"  # Path to the results directory
output_file = "Results.rst"  # Output .rst file
# Create an instance of RSTGenerator and generate the documentation
generator = RSTGenerator(results_dir, output_file)
generator.generate_rst()
print(f"Generated .rst documentation at: {output_file}")

extensions = []
html_logo = "_static/logo.png"

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinxawesome_theme"
html_static_path = ['_static']
