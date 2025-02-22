import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pandarallel import pandarallel
import folium
import os
#from deepface import DeepFace
from tqdm import tqdm
import clip
from PIL import Image
import torch

#packages for calculating spelling errors
import spacy
from textblob import TextBlob 
from bs4 import BeautifulSoup  

#packages for calculating aesthetic scores
import requests
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image  # Needed for image loading and preprocessing



print(tf.__version__)


##### This is a script to add additional custom features to the AirBnB data #####


class AddCustomFeatures:
    def __init__(
            self, 
            data: pd.DataFrame, 
            additional_features: list, 
            host_profile_picture_dir: str = "C:/Users/nilsk/Dokumente/Machine Learning (MSc.)/1. Semester/Data Literacy/oslo/host_picture_url",
            picture_url_dir: str = "C:/Users/nilsk/Dokumente/Machine Learning (MSc.)/1. Semester/Data Literacy/oslo/picture_url"):
        self.data = data
        self.features = []
        self.host_profile_picture_dir = host_profile_picture_dir
        self.picture_url_dir = picture_url_dir

        # Add centrality feature:
        if 'distance_to_city_center' in additional_features:
            try:
                print("Calculating centrality...")
                self.calculate_centrality()
            except Exception as e:
                print(f"Error calculating centrality: {e}")

        if 'review_sentiment' in additional_features:
            try:
                print("Adding review sentiment...")
                self.add_review_sentiment()
            except Exception as e:
                print(f"Error adding review sentiment: {e}")

        if 'average_review_length' in additional_features:
            try:
                print("Adding review length...")
                self.add_review_length()
            except Exception as e:
                print(f"Error adding review length: {e}")

        if 'spelling_errors' in additional_features:
            try:
                print("Adding spelling evaluation...")
                self.nlp = spacy.load("en_core_web_sm")
                self.add_spelling_evaluation()
            except Exception as e:
                print(f"Error adding spelling evaluation: {e}")
        
        if 'host_profile_analysis' in additional_features:
            try:
                print("Adding host profile analysis...")
                self.add_host_profile_analysis()
            except Exception as e:
                print(f"Error adding host profile analysis: {e}")
        
        if 'aesthetic_score' in additional_features:
            try:
                print("Adding aesthetic score...")
                self.load_pretrained_nima()
                self.add_aesthetic_score()
            except Exception as e:
                print(f"Error adding aesthetic score: {e}")

        if 'listing_picture_analysis' in additional_features:
            try:
                print("Adding listing picture analysis...")
                self.add_listing_picture_analysis()
            except Exception as e:
                print(f"Error adding listing picture analysis: {e}")


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
    
    def add_host_profile_analysis(self):

        assert len(self.data["city"].unique()) == 1, "only single city dataframes can be processed here"

        city = self.data["city"].iloc[0]
        n = len(self.data)
        host_picture_url_dir = self.host_profile_picture_dir
        print(host_picture_url_dir)
        dir_n = len(os.listdir(host_picture_url_dir))
        assert dir_n == n, f"number of pictures {dir_n} in directory {host_picture_url_dir}  must match number of listings {n}(len of df)"


        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        text = clip.tokenize(["professional photo", "casual photo", "blurry image"]).to(device)

        people_visible_column = []
        male_or_female_column = []
        setting_indoor_outdoor_column = []
        professionality_column = []
        quality_column = []

        people_visible = clip.tokenize(["person", "no person"]).to(device)
        male_or_female = clip.tokenize(["male", "female"]).to(device)
        setting_indoor_outdoor = clip.tokenize(["indoor", "outdoor"]).to(device)
        professionality = clip.tokenize(["professional photo", "casual photo"]).to(device)
        quality = clip.tokenize(["high quality photo", "low quality or blurry photo"]).to(device)

        people_visible_features = model.encode_text(people_visible)
        male_or_female_features = model.encode_text(male_or_female)
        setting_indoor_outdoor_features = model.encode_text(setting_indoor_outdoor)
        professionality_features = model.encode_text(professionality)
        quality_features = model.encode_text(quality)

        for i in tqdm(range(n)):
            image_path = host_picture_url_dir + f"/image_{i}.jpg"
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
                
                people_visible_similarity = (image_features @ people_visible_features.T).softmax(dim=-1)
                male_or_female_similarity = (image_features @ male_or_female_features.T).softmax(dim=-1)
                setting_indoor_outdoor_similarity = (image_features @ setting_indoor_outdoor_features.T).softmax(dim=-1)
                professionality_similarity = (image_features @ professionality_features.T).softmax(dim=-1)
                quality_similarity = (image_features @ quality_features.T).softmax(dim=-1)
            
            people_visible_score = people_visible_similarity[0][0].item()
            male_or_female_score = male_or_female_similarity[0][0].item()
            setting_indoor_outdoor_score = setting_indoor_outdoor_similarity[0][0].item()
            professionality_score = professionality_similarity[0][0].item()
            quality_score = quality_similarity[0][0].item()

            people_visible_column.append(people_visible_score)
            male_or_female_column.append(male_or_female_score)
            setting_indoor_outdoor_column.append(setting_indoor_outdoor_score)
            professionality_column.append(professionality_score)
            quality_column.append(quality_score)

        
        self.data['host_profile_pic_people_visible'] = people_visible_column
        self.data['host_profile_pic_male_or_female'] = male_or_female_column
        self.data['host_profile_pic_setting_indoor_outdoor'] = setting_indoor_outdoor_column
        self.data['host_profile_pic_professionality'] = professionality_column
        self.data['host_profile_pic_quality'] = quality_column
        
    def add_listing_picture_analysis(self):

        assert len(self.data["city"].unique()) == 1, "only single city dataframes can be processed here"

        city = self.data["city"].iloc[0]
        n = len(self.data)
        picture_url_dir = self.picture_url_dir
        print(picture_url_dir)
        dir_n = len(os.listdir(picture_url_dir))
        assert dir_n == n, f"number of pictures {dir_n} in directory {picture_url_dir}  must match number of listings {n}(len of df)"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        text = clip.tokenize(["professional photo", "casual photo", "blurry image"]).to(device)

        
        setting_indoor_outdoor_column = []
        setting_indoor_outdoor = clip.tokenize(["indoor", "outdoor"]).to(device)
        setting_indoor_outdoor_features = model.encode_text(setting_indoor_outdoor)

        for i in tqdm(range(n)):
            image_path = picture_url_dir + f"/image_{i}.jpg"
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
                
                setting_indoor_outdoor_similarity = (image_features @ setting_indoor_outdoor_features.T).softmax(dim=-1)
                
            setting_indoor_outdoor_score = setting_indoor_outdoor_similarity[0][0].item()
            setting_indoor_outdoor_column.append(setting_indoor_outdoor_score)
            
        self.data['picture_url_setting_indoor_outdoor'] = setting_indoor_outdoor_column

   # method to load a NIMA model to predict aesthetic scores
    def load_pretrained_nima(self):
        base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        print("MobileNetV2 model loaded successfully!")

        x = GlobalAveragePooling2D()(base_model.output)
        # hidden layer
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(10, activation='softmax')(x)

        self.aesthetic_model = Model(inputs=base_model.input, outputs=predictions)
        # load pretrained NIMA weights (MobilnetV2)
        weight_path = "./mobilenet_weights.h5" 

        print(f"Loading weights from: {weight_path}")
        try:
            self.aesthetic_model.load_weights(weight_path, by_name=True, skip_mismatch=True)
            print("Pre-trained NIMA weights loaded successfully!")
        except Exception as e:
            print(f"Error loading weights: {e}. Check if the path is correct.")
        
        self.aesthetic_model.compile(optimizer='adam', loss='categorical_crossentropy')
        self.aesthetic_model.trainable = False

    # method to predict aesthetic score of an image using the NIMA model
    def add_aesthetic_score(self):
        assert len(self.data["city"].unique()) == 1, "only single city dataframes can be processed here"

        city = self.data["city"].iloc[0]
        n = len(self.data)
        listing_picture_url_dir = self.picture_url_dir
        print(f"Loading listing pictures from: {listing_picture_url_dir}")
        dir_n = len(os.listdir(listing_picture_url_dir))
        assert dir_n == n, f"number of pictures {dir_n} in directory {listing_picture_url_dir}  must match number of listings {n}(len of df)"

        aesthetic_scores = []
        for i in tqdm(range(n)):

            # load listing picture from directory + preprocessing
            image_path = listing_picture_url_dir +  f"/image_{i}.jpg"
            img = Image.open(image_path).resize((224, 224), Image.LANCZOS)
            img = img.convert('RGB')
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array_preprocessed = preprocess_input(img_array)

            # model predictions
            predictions = self.aesthetic_model.predict(img_array_preprocessed)[0]

            # compute aesthetic score and standard deviation
            aesthetic_score = np.sum(predictions * np.arange(1, 11))
            variance = np.sum(predictions * (np.arange(1, 11) - aesthetic_score)**2)
            standard_deviation = np.sqrt(variance)
            print(f"Listing {i} → Aesthetic Score: {aesthetic_score:.2f} ± {standard_deviation:.2f}")
            
            aesthetic_scores.append(max(1.0, min(10.0, aesthetic_score)))
            
        self.data['aesthetic_score'] = aesthetic_scores    


    # Returns the data
    def return_data(self):
        return self.data

def main():
    dataframe = pd.DataFrame()
    for city in ['berlin', "barcelona", "istanbul", "london", "oslo"]:
    #    data = f'/home/sn/pCloudDrive/AirBnB_Daten/European_Cities/European_Cities_Preprocessed/{city}/single_city_listing.csv'
    #    data = pd.read_csv(data)
    #    additional_features = ['distance_to_city_center', 'spelling_errors', 'aesthetic_score', 'host_profile_analysis', 'listing_picture_analysis', 'average_review_length']
    #    data = AddCustomFeatures(data, additional_features, host_profile_picture_dir=f'/home/sn/pCloudDrive/AirBnB_Daten/European_Cities/European_Cities_Preprocessed/images/{city}/host_picture_url', picture_url_dir=f'/home/sn/pCloudDrive/AirBnB_Daten/European_Cities/European_Cities_Preprocessed/images/{city}/picture_url').return_data()
    #    data.to_csv(f'/home/sn/pCloudDrive/AirBnB_Daten/European_Cities/European_Cities_Preprocessed/{city}/{city}_data.csv', index=False)#

        city_folder = f'/home/sn/pCloudDrive/AirBnB_Daten/European_Cities/European_Cities_Preprocessed/{city}'
        data = pd.read_csv(f'{city_folder}/{city}_data.csv')
        dataframe = pd.concat([dataframe, data], ignore_index=True)

    dataframe.to_csv(f'/home/sn/pCloudDrive/AirBnB_Daten/European_Cities/European_Cities_Preprocessed/european_cities_data.csv', index=False)


if __name__ == "__main__":
    main()