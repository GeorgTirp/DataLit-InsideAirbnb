import numpy as np
import pandas as pd
import os
from copy import deepcopy
import torch 
import transformers as tf
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import sklearn
from sklearn.decomposition import PCA
from typing import Tuple, Dict 
import requests
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights

PATH_TO_REPO = "C:/Users/nilsk/Dokumente/Machine Learning (MSc.)/1. Semester/Data Literacy/DataLit-InsideAirbnb"
RAW_DATA_DIR = PATH_TO_REPO + '/data/raw_data'
SAVING_DIR = PATH_TO_REPO + '/data/preprocessed_data'
PROCESS_ALL_CITIES = True
CITY_LIST   = ["berlin"] #list cities which should be processed if not PROCESS_ALL_CITIES
DEBUG_MODE = False # determines if preprocessing is in DEBUG_MODE (no processing of file --> execution of main-function)



class InsideAirbnbDataset:
    def __init__(
            self,
            raw_data_dir: str = "/kaggle/input/berlin-amsterdam/raw_data",
            process_all_cities: bool = True,
            cities_to_process: list   = ["berlin"],
            read_from_raw = True,
            preprocessed_data_dir = 'preprocessed_data',
            file_name = 'single_city_listing.csv'):
        
        self.process_all_cities = process_all_cities
        self.cities_to_process = cities_to_process
        
        if read_from_raw:
            self.raw_data_dir = raw_data_dir
    
            # read in raw data from raw data directory in repository
            self.raw_data_dict = self._read_data_from_files()
    
            # integrate the reviews from reviews df into the listings df for each city in the raw_data_dict
            self._integrate_reviews_into_listings()
    
            # aggregate all listings dfs from each city and store in one all_cities_listings df
            self.all_cities_listings = self._aggregate_city_listings_into_one_df()
        else:
            self.preprocessed_data_dir = preprocessed_data_dir
            self.all_cities_listings = self._read_preprocessed_listings(file_name=file_name)
        
    
    def _read_data_from_files(self):
        print(f"reading in data from {self.raw_data_dir}")
        cities_in_raw_data_dir = os.listdir(self.raw_data_dir)

        if not self.process_all_cities and not set(self.cities_to_process).issubset(cities_in_raw_data_dir):
            raise ValueError("not all requested citys are in directory")
        
        raw_data_dict = {}

        if self.process_all_cities:
            self.cities_to_process = cities_in_raw_data_dir

        self.cities = self.cities_to_process
        
        for city in self.cities_to_process:
            print(f"collecting data for city: {city}")
            raw_data_dict[city] = {}
            city_dir = self.raw_data_dir + '/' + city
            file_names = [f for f in os.listdir(city_dir) if os.path.isfile(os.path.join(city_dir, f))]

            for file_name in file_names:
                if file_name.endswith('.csv') or file_name.endswith('.geojson') or file_name.endswith('.csv.gz'):
                    file_path = os.path.join(city_dir, file_name)
            
                    # Read the file into a DataFrame
                    if file_name.endswith('.geojson'):
                        df = pd.read_json(file_path)  # Adjust based on the specific geojson handling
                    else:
                        file_name_core = file_name.split(sep=".")[0]

                        if file_name_core == "reviews":
                            index_col = 1
                        else:
                            index_col = 0
                            
                        df = pd.read_csv(file_path, index_col=index_col)

                        # filter out all listings which do not have price available
                        if file_name_core == "listings":
                            df = df[df["price"].notna()]

                    raw_data_dict[city][file_name] = df

        print(f"collecting data process done")

        return raw_data_dict

    def _integrate_reviews_into_listings(self):
        print(f"initializing reviews collection process and integration into city listings")
        
        for city in self.cities:
            print(f"current city: {city}")
            city_listings = self.raw_data_dict[city]["listings.csv"]
            city_reviews = self.raw_data_dict[city]["reviews.csv"]       
            city_calendar = self.raw_data_dict[city]["calendar.csv"] 

            city_listings_indices = city_listings.index.to_list()
            city_listings["comments"] = [[] for _ in range(len(city_listings))]

            for index in city_listings_indices:
                city_index_reviews = city_reviews[city_reviews["listing_id"] == index]
                comments = city_index_reviews["comments"].to_list()

                comments_with_newline = []
                for comment in comments:
                    if type(comment) is float: #if it is nan, as nan are float values
                        comment = ""
                    comment_transformed = comment.replace('<br/>', '\n').replace('\r', '')
                    comments_with_newline.append(comment_transformed)

                city_listings.at[index, 'comments'] = comments_with_newline
        
        print("integration of reviews into cites listings done")

    def _aggregate_city_listings_into_one_df(self):
        print("initializing aggregation of regional listings into one dataframe")
        all_cities_listings = []

        for city in self.cities:
            city_listings = self.raw_data_dict[city]["listings.csv"]
            city_listings.insert(0, 'city', city)
            all_cities_listings.append(city_listings)

        all_cities_listings = pd.concat(all_cities_listings, ignore_index=True)
        print("aggregation done")
        return all_cities_listings

    def filter_listings_and_impute_nan(self,
                                       meta_data_columns = [
                                           'listing_url', 
                                           'host_location', 
                                           'scrape_id', 
                                           'last_scraped', 
                                           'source',  
                                           'host_id', 
                                           'host_url', 
                                           'host_name', 
                                           'host_neighbourhood',
                                           'host_thumbnail_url', 
                                           'host_verifications', 
                                           'neighbourhood', 
                                           'neighbourhood_group_cleansed', 
                                           'calendar_last_scraped', 
                                           'license'
                                       ],
                                       nan_columns = ['calendar_updated'],
                                       include_only_reviewed = True):
        
        all_cities_listings = self.all_cities_listings
        #filter the NaN columns (here are no entries)
        all_cities_listings = all_cities_listings.drop(columns = nan_columns)
        
        #filter the meta_data columns (these columns are not used for prediction)
        all_cities_listings = all_cities_listings.drop(columns = meta_data_columns)
    

        #categorical NaN:
        #  'host_response_time' --> extra NaN category
        all_cities_listings['host_response_time'].fillna(value='not available', inplace=True)
    
        #  'host_is_superhost'  --> False
        all_cities_listings['host_is_superhost'].fillna(value='f', inplace=True)
        
        #  'has_availability'   --> False
        all_cities_listings['has_availability'].fillna(value='f', inplace=True)

        
        #numerical NaN:
        #  'host_response_rate'    --> mean/median
        all_cities_listings['host_response_rate'] = all_cities_listings['host_response_rate'].map(
            lambda s: s.removesuffix('%') if type(s) == str else s, 
            na_action = 'ignore'
        )
        median_value = all_cities_listings['host_response_rate'].median()
        all_cities_listings['host_response_rate'].fillna(value=median_value, inplace=True)

        #  'host_acceptance_rate'  --> mean/median
        all_cities_listings['host_acceptance_rate'] = all_cities_listings['host_acceptance_rate'].map(
            lambda s: s.removesuffix('%') if type(s) == str else s, 
            na_action = 'ignore'
        )
        median_value = all_cities_listings['host_acceptance_rate'].median()
        all_cities_listings['host_acceptance_rate'].fillna(value=median_value, inplace=True)

        #  'bathrooms'        --> mean/median
        median_value = all_cities_listings['bathrooms'].median()
        all_cities_listings['bathrooms'].fillna(value=median_value, inplace=True)

        #  'bedrooms'         --> mean/median
        median_value = all_cities_listings['bedrooms'].median()
        all_cities_listings['bedrooms'].fillna(value=median_value, inplace=True)

        #  'beds'         --> mean/median
        median_value = all_cities_listings['beds'].median()
        all_cities_listings['beds'].fillna(value=median_value, inplace=True)

        #  'first_review'     --> leave out listings without reviews (only include listings which include reviews which indicates that listings are booked and price is valid)
        #   .....
        #  'reviews_per_mon'  --> leave out listings without reviews (only include listings which include reviews which indicates that listings are booked and price is valid)
        review_columns = [
                'first_review', 
                'review_scores_rating', 
                'review_scores_accuracy', 
                'review_scores_cleanliness', 
                'review_scores_checkin',
                'review_scores_communication', 
                'review_scores_location', 
                'review_scores_value', 
                'reviews_per_month'
            ]
        if include_only_reviewed:
            all_cities_listings = all_cities_listings[all_cities_listings['first_review'].notna()]
            for review_col in review_columns:
                assert not all_cities_listings[review_col].isna().any()
        else:
            # since I could not think of an sensible imputation strategy
            all_cities_listings.drop(columns = review_columns)
        
        #NLP NaN:
        #  'name'                   --> embedding of empty string ""
        #  'description'            --> embedding of empty string ""
        #  'neighborhood_overview'  --> embedding of empty string ""
        #  'host_about'             --> embedding of empty string ""
        # --> these are handled in the add_nlp_embedding function
    
        #  'comments'             --> embedding of empty string ""
        # --> handled in the _integrate_reviews_into_listings function

        self.all_cities_listings = all_cities_listings
        
    
    def add_nlp_embedding(self, 
                          nlp_col_names = ['name', 'description', 'neighborhood_overview', 'host_about', 'amenities','comments'], 
                          batch_size = 32):
        print("initializing NLP embedding process")
        print(f"batch size: {batch_size}") 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_name = 'distilbert-base-multilingual-cased'
        tokenizer = tf.AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        model = tf.AutoModel.from_pretrained(model_name).to(device)
        print(f"embeddings are computed using transformer model: {model_name} from hugging face")
        
        for nlp_col_name in nlp_col_names:
            print(f"current nlp column: {nlp_col_name}")

            nlp_col = self.all_cities_listings[nlp_col_name]
            nlp_col_list = []

            # convert nlp columns to a list 
            if nlp_col_name in ['name', 'description', 'neighborhood_overview', 'host_about', 'comments']:
                nlp_col_list = nlp_col.fillna(value="").to_list()
            elif nlp_col_name == "amenities":
                for amenities_raw_entry in nlp_col:
                    amenities_collection = json.loads(amenities_raw_entry) # amenities_raw_entry is in json string format
                    nlp_col_list.append(amenities_collection)
            else:
                raise ValueError(f"no procedure found for converting {nlp_col_name} to list")
            

            nlp_col_list_embedded = []

            pooling_approach = ['amenities', 'comments']
            # for each entry in nlp column, single embeddings are inferred for amenity_items / single reviews --> then mean pooling
            if nlp_col_name in pooling_approach:
                for i, entry in enumerate(tqdm(nlp_col_list)):
                    if entry == []:
                        entry = np.asarray([" "])
                        
                    dataloader = DataLoader(entry, batch_size=batch_size)
                    entry_items_embeddings_list = []
                    
                    for batch in dataloader:
                        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
                        with torch.no_grad():
                            outputs = model(**inputs)
                        embeddings = outputs.last_hidden_state[:, 0, :]
                        embeddings = embeddings.squeeze(0).cpu().numpy()
                        entry_items_embeddings_list.append(embeddings)
                    
                    embeddings_array = np.vstack(entry_items_embeddings_list)
                    mean_pooled_embedding = np.mean(embeddings_array, axis=0)
                    nlp_col_list_embedded.append(mean_pooled_embedding)
                    
            # embeddings are inferred directly for the entries of all other nlp columns
            else:
                dataloader = DataLoader(nlp_col_list, batch_size=batch_size)
                elements = 0
                for batch in tqdm(dataloader):
                    elements += len(batch)
                    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :]
                    embeddings = embeddings.cpu().numpy()
                    if len(list(embeddings)) != len(batch):
                        print(embeddings.shape)
                    nlp_col_list_embedded += list(embeddings)
            
            nlp_col_embedded_name = nlp_col_name + '_emb'
            self.all_cities_listings[nlp_col_embedded_name] = nlp_col_list_embedded
        
        print("nlp embedding done")
    
    def dimensionality_reduction(self, 
                                 col_names = [
                                    'name_emb', 
                                    'description_emb', 
                                    'neighborhood_overview_emb', 
                                    'host_about_emb', 
                                    'amenities_emb',
                                    'comments_emb',
                                    'host_picture_emb',
                                    'picture_emb'
                                 ],
                                keep_variance = 0.95):
        
        print("initializing dimensionality reduction")
            
        for col_name in col_names:
            print(f"current embeddings: {col_name}")
            col = self.all_cities_listings[col_name]
            col_array = np.asarray([np.asarray(entry) for entry in col])

            pca = PCA(n_components = keep_variance, svd_solver='full')
            pca.fit(col_array)
            dim_red_col_array = pca.transform(col_array)
            print(f"used {pca.n_components_ } components for dim reduction to explain {keep_variance*100}% of the data")
            
            dim_red_col_name = col_name + '_dim_red'
            self.all_cities_listings[dim_red_col_name] = list(dim_red_col_array)
        print("dimensionality reduction done")

    def download_images_and_save(self, 
                            image_url_col_names = ['host_picture_url','picture_url'], 
                            saving_dir = 'kaggle/working/preprocessed_data/filtered_dataset_images',
                            process_n_images = -1):
        
        print("initializing image embedding process")
        for city in self.cities:
            for image_url_col_name in image_url_col_names:
                print(f"downloading images from web for column '{image_url_col_name}'")

                city_listings = self.all_cities_listings[self.all_cities_listings["city"] == city]
                image_url_col = city_listings[image_url_col_name]
                image_list = []
                no_access_indices = []
                image_size = (256,256)
                
                for i, image_url in enumerate(tqdm(image_url_col)):
                    if process_n_images >= 0 and i == process_n_images:
                        break
                    response = requests.get(image_url)
                    
                    # NaN values are floats
                    if type(image_url) is float:
                        no_access_indices.append(i)
                        image_list.append(Image.new("RGB", image_size))
                    else:
                        response = requests.get(image_url)
                        # code for successful request is 200
                        if response.status_code == 200:
                            try:
                                image = Image.open(BytesIO(response.content)).resize(image_size)
                                if image.mode != "RGB":
                                    image = image.convert('RGB')
                                image_list.append(image)
                            except OSError as e:
                                no_access_indices.append(i)
                                image_list.append(Image.new("RGB", image_size))  
                        else:
                            no_access_indices.append(i)
                            image_list.append(Image.new("RGB", image_size))
                            #response.raise_for_status()
        
                print(f"pictures from rows {no_access_indices} could not be accessed")

                image_saving_path = saving_dir + '/' + city + '/' + image_url_col_name
                if not os.path.exists(image_saving_path):
                    os.makedirs(image_saving_path)
                for i, image in enumerate(image_list):
                    image.save(image_saving_path + '/' + f"image_{i}.jpg")
                
            

    def add_image_embedding(self, 
                            image_url_col_names = ['host_picture_url','picture_url'], 
                            batch_size = 32,
                            read_from_dir = False,
                            read_dir = 'kaggle/working/preprocessed_data/filtered_dataset_images',
                            embedd_n_images = -1):
        
        print("initializing image embedding process")
        
        for image_url_col_name in image_url_col_names:
            
            if read_from_dir:
                print(f"reading images from directory for column '{image_url_col_name}'")
                
                cities_subdirectories = [d for d in os.listdir(read_dir) if os.path.isdir(os.path.join(read_dir, d))]

                if not set(set(self.cities)).issubset(cities_subdirectories):
                    raise ValueError("not all cities in need to be processed are in given read directory")

                for city in self.cities:
                    column_city_subdirectory = read_dir + '/' + city + '/' + image_url_col_name
                    assert os.path.exists(column_city_subdirectory), f"subdirectory {column_city_subdirectory} does not exist"
                    image_files = os.listdir(column_city_subdirectory)
                    assert len(image_files) == len(self.all_cities_listings)

                    image_list = []
                    for image_file in image_files:
                        image = Image.open(os.path.join(column_city_subdirectory, image_file))
                        image_list.append(image)

                    assert len(image_list) == len(self.all_cities_listings)
                    
            else:
                print(f"downloading images from web for column '{image_url_col_name}'")
                image_url_col = self.all_cities_listings[image_url_col_name]
                image_list = []
                no_access_indices = []
                image_size = (256,256)
                
                for i, image_url in enumerate(tqdm(image_url_col)):
                    if embedd_n_images >= 0 and i == embedd_n_images:
                        break
                    response = requests.get(image_url)
                    
                    # NaN values are floats
                    if type(image_url) is float:
                        no_access_indices.append(i)
                        image_list.append(Image.new("RGB", image_size))
                    else:
                        response = requests.get(image_url)
                        # code for successful request is 200
                        if response.status_code == 200:
                            try:
                                image = Image.open(BytesIO(response.content)).resize(image_size)
                                if image.mode != "RGB":
                                    image = image.convert('RGB')
                                image_list.append(image)
                            except OSError as e:
                                no_access_indices.append(i)
                                image_list.append(Image.new("RGB", image_size))  
                        else:
                            no_access_indices.append(i)
                            image_list.append(Image.new("RGB", image_size))
                            #response.raise_for_status()
        
                print(f"pictures from rows {no_access_indices} could not be accessed")
            print("transform images and construct dataloader")
    
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            
            image_transform = transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            normalize
                                            ])
            tensor_image_list = [image_transform(image) for image in image_list]
    
            data_loader = DataLoader(tensor_image_list, batch_size=batch_size)
            
            resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            modules = list(resnet.children())[:-1]  # remove the FC layer
            resnet_feature_extractor = torch.nn.Sequential(*modules)
            resnet_feature_extractor.eval()
            
            print("embedding image data using ResNet50")
            feature_embeddings_list = []
       
            for batch in tqdm(data_loader):
                with torch.no_grad():
                    feature_embeddings = resnet_feature_extractor(batch)
                feature_embeddings = feature_embeddings.view(feature_embeddings.size(0), -1).numpy()
    
                feature_embeddings_list += list(feature_embeddings)
    
            col_name_core = image_url_col_name.split('_')[:-1]
            image_col_embedded_name = '_'.join(col_name_core + ['emb'])
            
            # only important if embedd_n_images not -1 --> not all images get embedded
            feature_embeddings_list_n = len(feature_embeddings_list)
            all_listings_n = len(self.all_cities_listings)
            diff = all_listings_n - feature_embeddings_list_n
            for _ in range (diff):
                feature_embeddings_list.append([]) 
                
    
            valid_feature_embeddings_list = deepcopy(feature_embeddings_list)[:feature_embeddings_list_n]
            for index in no_access_indices[::-1]:
                del valid_feature_embeddings_list[index]
    
            valid_feature_embeddings_array = np.asarray(valid_feature_embeddings_list)
            mean_embedding = np.mean(valid_feature_embeddings_array, axis=0)
            print(f"mean_embedding: {mean_embedding}")
            
            for no_access_index in no_access_indices:
                feature_embeddings_list[no_access_index] = mean_embedding
    
            self.all_cities_listings[image_col_embedded_name] = feature_embeddings_list
            
        print("image embedding done")
    
    def save_all_cities_listings_to_file(self, 
                                         file_name = "single_city_listing.csv", 
                                         saving_dir =  'preprocessed_data',
                                         single_data_frames = True):
        
        self.saving_dir = saving_dir

        if single_data_frames:
            cities = self.all_cities_listings["city"].unique()
            for city in cities:
                city_listings = self.all_cities_listings[self.all_cities_listings["city"] == city]
                city_dir = saving_dir + '/' + f"/{city}"
                
                if not os.path.exists(city_dir):
                    os.makedirs(city_dir)
                file_path = city_dir + '/' + file_name 
                
                city_listings.to_csv(file_path)
                print(f"{city} listings saved to path: {file_path}")
        else:
            file_path = saving_dir + '/' + file_name 
            self.all_cities_listings.to_csv(file_path)
            print(f"all cities listings saved to path: {file_path}")

    def _read_preprocessed_listings(self, file_name):
        
        cities_subdirectories = [d for d in os.listdir(self.preprocessed_data_dir) if os.path.isdir(os.path.join(self.preprocessed_data_dir, d))]

        
        if not set(set(self.cities_to_process)).issubset(cities_subdirectories):
            raise ValueError("not all cities in need to be processed are in given read directory")

        if self.process_all_cities:
            self.cities_to_process = cities_subdirectories
        self.cities = self.cities_to_process
        all_cities_listings = []

        for city in self.cities:
            print(f"collecting data for city: {city}")
            city_dir = self.preprocessed_data_dir + '/' + f"{city}"
            city_dir_content = os.listdir(city_dir)
            
            assert len(city_dir_content) == 1, f"there must exist only one single listing for city {city} in directory {city_dir}"
            city_file = city_dir_content[0]
            file_path = city_dir + '/' + file_name 
            
            city_listings = pd.read_csv(file_path, index_col=0)
            all_cities_listings.append(city_listings)
            
        all_cities_listings = pd.concat(all_cities_listings, ignore_index=True)
        print("reading preprocessed cities from directoy done")
        return all_cities_listings
        

def main():
    data_set = InsideAirbnbDataset()
    data_set.save_all_cities_listings_to_file('ignore_all_listings.csv')
    data_set.add_nlp_embedding(nlp_col_names = ['name'])
    data_set.dimensionality_reduction(col_names = ['name_emb'])


if __name__ == "__main__":
    if not DEBUG_MODE:
        data_set = InsideAirbnbDataset()

    else:
        main()

    









