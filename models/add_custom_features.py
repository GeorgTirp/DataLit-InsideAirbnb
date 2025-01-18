import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pandarallel import pandarallel





class AddCustomFeatures:
    def __init__(self, data, additional_features: list):
        self.data = data
        self.features = []

        # Add centrality feature:
        if 'distance_to_city_center' in additional_features:
            self.calculate_centrality()

        if 'review_sentiment' in additional_features:
            self.add_review_sentiment()

    def calculate_centrality(self):
        central_lat = self.data['latitude'].mean()
        central_lon = self.data['longitude'].mean()
        self.data['distance_to_city_center'] = np.sqrt((self.data['latitude'] - central_lat)**2 + (self.data['longitude'] - central_lon)**2)

    def add_review_sentiment(self):
        pandarallel.initialize(progress_bar=True)
        self.data['sentiment_score'] = self.data['comments'].parallel_apply(self.analyze_sentiment)


    # Function to compute sentiment
    def analyze_sentiment(self, text):
        analyzer = SentimentIntensityAnalyzer()
        return analyzer.polarity_scores(text)['compound']
    

    def return_data(self):
        return self.data
    

