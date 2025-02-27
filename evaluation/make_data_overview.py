import pandas as pd
import sweetviz as sv



df = pd.read_csv('/media/sn/Frieder_Data/Master_Machine_Learning/data_preprocessed/european_cities_data_-40000_london.csv')


sv.config_parser.read("Override.ini")

# Generate the report
report = sv.analyze(df)

# Save the report as an HTML file
report.show_html("sweetviz_report.html")