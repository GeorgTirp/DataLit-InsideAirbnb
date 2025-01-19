import os
from flask import send_file
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import plotly.graph_objects as go

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # Access the Flask server instance

# Set folder path for files
FOLDER_PATH = '/home/frieder/Desktop/Uni/Machine_Learning/DataLit/DataLit-InsideAirbnb/results/test_run'

# Get lists of files
graphs = [f for f in os.listdir(FOLDER_PATH) if f.endswith('.png')]
logs = [f for f in os.listdir(FOLDER_PATH) if f.endswith('.txt') or f.endswith('.log')]
predictions = [f for f in os.listdir(FOLDER_PATH) if f.endswith('.csv')]

# Set defaults (first file in each category if available)
default_graph = graphs[0] if graphs else None
default_log = logs[0] if logs else None
default_prediction = predictions[0] if predictions else None

# Layout
app.layout = html.Div([
    html.H1("Pipeline Results Visualization"),

    # Prediction File Selection
    html.Div([
        html.Label("Select a Predictions File:"),
        dcc.Dropdown(
            id='prediction-dropdown',
            options=[{'label': p, 'value': p} for p in predictions],
            value=default_prediction,  # Pre-select default
            placeholder="Select predictions"
        ),
        dcc.Graph(id='prediction-graph'),
    ]),
    # Graph Selection
    html.Div([
        html.Label("Select a Graph:"),
        dcc.Dropdown(
            id='graph-dropdown',
            options=[{'label': g, 'value': g} for g in graphs],
            value=default_graph,  # Pre-select default
            placeholder="Select a graph"
        ),
        html.Img(id='graph-display', style={'max-width': '80%', 'max-height': '600px'}),
    ], style={'margin-bottom': '20px'}),

    # Log File Selection
    html.Div([
        html.Label("Select a Log File:"),
        dcc.Dropdown(
            id='log-dropdown',
            options=[{'label': l, 'value': l} for l in logs],
            value=default_log,  # Pre-select default
            placeholder="Select a log file"
        ),
        html.Pre(id='log-display', style={
            'border': '1px solid #ccc',
            'padding': '10px',
            'whiteSpace': 'pre-wrap',
            'height': '300px',
            'overflowY': 'scroll'
        }),
    ], style={'margin-bottom': '20px'}),
])

# Flask route to serve images
@app.server.route('/images/<path:path>')
def serve_image(path):
    """Serve image files from the local folder."""
    file_path = os.path.join(FOLDER_PATH, path)
    if os.path.exists(file_path):
        return send_file(file_path)
    else:
        return f"Error: File {file_path} not found.", 404

# Callback to update the image
@app.callback(
    Output('graph-display', 'src'),
    Input('graph-dropdown', 'value')
)
def update_graph(graph_file):
    if graph_file:
        return f'/images/{graph_file}'  # This will trigger the Flask route
    return None

# Callback to display log content
@app.callback(
    Output('log-display', 'children'),
    Input('log-dropdown', 'value')
)
def update_log(log_file):
    if log_file:
        file_path = os.path.join(FOLDER_PATH, log_file)
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except Exception as e:
            return f"Error reading log file: {e}"
    return ""

# Callback to generate predictions graph
@app.callback(
    Output('prediction-graph', 'figure'),
    Input('prediction-dropdown', 'value')
)
def update_prediction_graph(pred_file):
    if pred_file:
        file_path = os.path.join(FOLDER_PATH, pred_file)
        try:
            # Load the predictions file
            df = pd.read_csv(file_path)
            if 'y_test' not in df.columns or 'y_pred' not in df.columns:
                return go.Figure().update_layout(title="Error: CSV must contain 'y_test' and 'y_pred' columns.")

            # Create a scatter plot
            fig = go.Figure()
            # Calculate R^2 score and Pearson correlation coefficient
            r2, p_val = pearsonr(df['y_test'], df['y_pred'])

            # Create a scatter plot
            fig.add_trace(go.Scatter(x=df['y_test'], y=df['y_pred'], mode='markers', name='Predictions'))
            fig.add_trace(go.Line(x=df['y_test'], y=df['y_test'], name='Ideal', line=dict(color='red', dash='dash')))
            fig.update_layout(
                title=f"True vs y_pred Values (R^2: {r2:.2f}, P-value: {p_val:.2f})",
                xaxis_title="y_test",
                yaxis_title="y_pred",
                legend_title="Legend"
            )
            return fig
        except Exception as e:
            return go.Figure().update_layout(title=f"Error loading predictions file: {e}")
    return go.Figure()

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
