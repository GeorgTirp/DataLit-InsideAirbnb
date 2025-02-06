import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pandarallel import pandarallel
import folium

#packages for calculating spelling errors
import spacy
from textblob import TextBlob 
from bs4 import BeautifulSoup  

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

        if 'spelling_errors' in additional_features:
            self.nlp = spacy.load("en_core_web_sm")
            self.add_spelling_evaluation()


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
        pandarallel.initialize(progress_bar=True, nb_workers=10)
        self.data['sentiment_score'] = self.data['comments'].parallel_apply(self.analyze_sentiment)
        # Positive / positive + negative


    # Function to compute sentiment for a list of reviews (compound score)
    def analyze_sentiment(self, text):
        analyzer = SentimentIntensityAnalyzer()
        if not text:  # Handle empty list case
            return 0.0  
        scores = [analyzer.polarity_scores(text)['compound'] for text in text]
        return sum(scores) / len(scores)  # Calculate average
    
    
    def add_review_length(self):
        def calculate_review_length(reviews):
            lengths = []
            for review in eval(reviews):
                lengths.append(len(review.split()))
            return np.mean(lengths)
        pandarallel.initialize(progress_bar=True)
        self.data['average_review_length'] = self.data['comments'].parallel_apply(calculate_review_length)

    def calculate_spelling_errors(self, description):
        
        #ignore list - current method to ignore ordinal numbers (from 1st to 1000th)
        ignore = [f"{i}{'st' if i % 10 == 1 and i % 100 != 11 else 'nd' if i % 10 == 2 and i % 100 != 12 else 'rd' if i % 10 == 3 and i % 100 != 13 else 'th'}" for i in range(1, 1001)]

        if pd.isna(description) or description == "": #check if description is empty (NaN values are already preprocessed but just in case)
                return np.nan
        
        # remove html tags from description
        clean_description = BeautifulSoup(description, "html.parser").get_text()
        
        nlp_description = self.nlp(clean_description)
        spelling_errors = 0

        # check description word for word 
        for token in nlp_description: 

            # makes sure to not flag GPE = geopolitical entity, LOC = location as spelling errors
            word = token.text
            if token.ent_type_ in ["GPE", "LOC"] or word in ignore: 
                continue
            
            # check for possible spelling errors
            corrected = TextBlob(word).correct() 
            if word.lower() != corrected.lower(): 
                spelling_errors += 1
            
        return spelling_errors / len(TextBlob(clean_description).words) #return ratio of spelling errors to total words
        

    def add_spelling_evaluation(self):
        pandarallel.initialize(progress_bar=True)
        self.data['spelling_errors'] = self.data['description'].parallel_apply(lambda x: self.calculate_spelling_errors(x))


    # Returns the data
    def return_data(self):
        return self.data
    



data = pd.read_csv('/home/frieder/pCloudDrive/AirBnB_Daten/Preprocessed_data/germany_preprocessed/berlin/city_listings.csv')
add_custom_features = ['distance_to_city_center', 'review_sentiment', 'average_review_length', 'spelling_errors']
features = AddCustomFeatures(data, add_custom_features)
data = features.return_data()
data.to_csv('/home/frieder/pCloudDrive/AirBnB_Daten/Preprocessed_data/germany_preprocessed/berlin/city_listings_custom_features.csv', index=False)