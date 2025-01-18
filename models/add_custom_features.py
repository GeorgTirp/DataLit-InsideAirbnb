import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pandarallel import pandarallel
import folium


##### This is a script to add additional custom features to the AirBnB data #####


class AddCustomFeatures:
    def __init__(self, data, additional_features: list):
        self.data = data
        self.features = []

        # Add centrality feature:
        if 'distance_to_city_center' in additional_features:
            self.calculate_centrality()

        if 'review_sentiment' in additional_features:
            self.add_review_sentiment()

        if 'average_review_length' in additional_features:
            self.add_review_length()


    # Calculates the distance from the middle of all the listing (e.g. city center) for each listing as measure of inverse centrality
    def calculate_centrality(self):
        central_lat = self.data['latitude'].mean()
        central_lon = self.data['longitude'].mean()
        
        # Plot the city center on a map to verify the coordinates
        def show_city_center_on_map(lat, lon):
            map_berlin = folium.Map(location=[lat, lon], zoom_start=12)
            folium.Marker([lat, lon], popup='City Center').add_to(map_berlin)
            return map_berlin

        # Show the city center on a map of Berlin
        city_center_map = show_city_center_on_map(central_lat, central_lon)
        city_center_map.save('city_center_map.html')

        # Calculate the distance to the city center and append to data
        self.data['distance_to_city_center'] = np.sqrt((self.data['latitude'] - central_lat)**2 + (self.data['longitude'] - central_lon)**2)


    # Add a sentiment score to the reviews - parallelized to utilize all cores (still takes about 1h for 10k reviews)
    def add_review_sentiment(self):
        pandarallel.initialize(progress_bar=True)
        self.data['sentiment_score'] = self.data['comments'].parallel_apply(self.analyze_sentiment)


    # Function to compute sentiment for a list of reviews (compound score)
    def analyze_sentiment(self, text):
        analyzer = SentimentIntensityAnalyzer()
        return analyzer.polarity_scores(text)['compound']
    
    
    def add_review_length(self):
        def calculate_review_length(reviews):
            lengths = []
            for review in eval(reviews):
                lengths.append(len(review.split()))
            return np.mean(lengths)
        pandarallel.initialize(progress_bar=True)
        self.data['average_review_length'] = self.data['comments'].parallel_apply(calculate_review_length)

    # Returns the data
    def return_data(self):
        return self.data
    

