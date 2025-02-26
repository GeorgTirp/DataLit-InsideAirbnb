import pandas as pd
from pretty_html_table import build_table
import sweetviz as sv
from ydata_profiling import ProfileReport


# Load your DataFrame
df = pd.read_csv('/media/sn/Frieder_Data/Master_Machine_Learning/data_preprocessed/european_cities_data.csv')


report = ProfileReport(df, explorative=True)
report.to_file("source/_static/european_cities_report.html")

