from models.regression import RegressionModels
from preprocessing.preprocessing_test import preprocess_data
from typing import Tuple, Dict
import pandas as pd
from fpdf import FPDF
import matplotlib.pyplot as plt
# Setting list of models
model_names = [
    "Linear Regression",
    "Random Forest Regression"
]

# Load all models
path = '/Users/georgtirpitz/Documents/Data_Literacy/'
data_path = path + 'example_data'
data_df = preprocess_data(data_path)
model = RegressionModels(data_df)
model.fit()
metrics = model.evaluate()
linear_features, random_forest_features = model.feature_importance(10)


def concat_evals(model_names:list, metrics: dict) -> dict:
    """Concetenates the evaluations of all models"""
    if len(model_names) != len(metrics):
        raise ValueError("Inputs do not match in length")
    evaluations = {}
    for name, metric in zip(model_names, metrics):
        evaluations[name] = metric
    return evaluations


def eval_table(evaluations: dict) -> None:
    """Makes a table with all the evaluation metrics"""
    # Create a DataFrame from the evaluations dictionary
    df = pd.DataFrame(evaluations)
    
    
    # Create a PDF class
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Model Evaluation Metrics', 0, 1, 'C')

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

        def chapter_title(self, title):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, title, 0, 1, 'L')
            self.ln(10)

        def chapter_body(self, body):
            self.set_font('Arial', '', 12)
            self.multi_cell(0, 10, body)
            self.ln()

        def add_table(self, df):
            self.set_font('Arial', '', 10)
            col_width = self.w / (len(df.columns) + 1)
            row_height = self.font_size * 1.5

            # Add table header
            for col in df.columns:
                self.cell(col_width, row_height, str(col), border=1)
            self.ln(row_height)

            # Add table rows
            for row in df.itertuples():
                for item in row[1:]:
                    self.cell(col_width, row_height, str(item), border=1)
                self.ln(row_height)

    # Create a PDF object
    pdf = PDF()
    pdf.add_page()

    # Add table to PDF
    pdf.add_table(df)
    
    # Save the PDF to a file
    pdf.output(path + 'graphs/evaluation_metrics.pdf')


def feature_hist(features: Dict[str, float], filename: str) -> None:
    """Plots a histogram of feature importances and saves it as a PDF"""
    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(features.keys(), features.values(), color='skyblue')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # Save the plot as a PDF
    plt.savefig(filename, format='pdf')
    plt.close()

    


if __name__ == "__main__":
    evaluations = concat_evals(model_names, metrics)
    eval_table(evaluations)
    feature_hist(linear_features, path + '/graphs/linear_features_importance.pdf')
    feature_hist(random_forest_features, path + '/graphs/random_forest_features_importance.pdf')