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
import logging
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import sys 
import gc
import asyncio
import aiohttp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class ImageDownloader:
    def __init__(self, cities, all_cities_listings, batch_size=500):
        self.cities = cities
        self.all_cities_listings = all_cities_listings
        self.image_size = (256, 256)
        self.batch_size = batch_size  # Process images in batches

    async def fetch_image(self, session, image_url):
        """Asynchronously fetch an image from a URL"""
        if isinstance(image_url, float):  # Handle NaN URLs
            return None

        try:
            async with session.get(image_url, timeout=120) as response:
                if response.status == 200:
                    return await response.read()  # Return raw image bytes
        except Exception as e:
            logging.warning(f"Failed to download {image_url}: {e}")
        return None

    async def download_images_async(self, image_urls):
        """Download images asynchronously in batches"""
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_image(session, url) for url in image_urls]
            return await asyncio.gather(*tasks)  # Fetch all images in parallel

    def process_and_save_image(self, img_bytes, image_path):
        """Process and save an image"""
        try:
            if img_bytes is None:
                img = Image.new("RGB", self.image_size)  # Placeholder image
            else:
                img = Image.open(BytesIO(img_bytes)).resize(self.image_size)
                if img.mode != "RGB":
                    img = img.convert("RGB")

            img.save(image_path, "JPEG")
        except Exception as e:
            logging.warning(f"Error processing image {image_path}: {e}")

    def download_images_and_save(self, 
                                 image_url_col_names=['host_picture_url', 'picture_url'], 
                                 saving_dir='kaggle/working/preprocessed_data/filtered_dataset_images', 
                                 process_n_images=-1):
        logging.info("Initializing image downloading process")

        for city in self.cities:
            for image_url_col_name in image_url_col_names:
                logging.info(f"Downloading images from column '{image_url_col_name}' for city '{city}'")

                city_listings = self.all_cities_listings[self.all_cities_listings["city"] == city]
                image_urls = city_listings[image_url_col_name].tolist()

                if process_n_images > 0:
                    image_urls = image_urls[:process_n_images]  # Limit number of images

                image_saving_path = os.path.join(saving_dir, city, image_url_col_name)
                os.makedirs(image_saving_path, exist_ok=True)

                # Process images in batches
                for batch_start in range(0, len(image_urls), self.batch_size):
                    batch_urls = image_urls[batch_start: batch_start + self.batch_size]
                    
                    logging.info(f"Processing batch {batch_start // self.batch_size + 1}, size: {len(batch_urls)}")

                    image_bytes_list = asyncio.run(self.download_images_async(batch_urls))

                    with ThreadPoolExecutor() as executor:
                        tasks = [
                            executor.submit(self.process_and_save_image, img_bytes, 
                                            os.path.join(image_saving_path, f"image_{batch_start + i}.jpg"))
                            for i, img_bytes in enumerate(image_bytes_list)
                        ]
                        for task in tqdm(tasks, desc=f"Processing batch {batch_start // self.batch_size + 1}"):
                            task.result()  # Wait for all tasks to complete

                    # Free up memory after each batch
                    del image_bytes_list
                    gc.collect()

                logging.info(f"Finished downloading images for {city} - {image_url_col_name}")



class InsideAirbnbDataset:
    """
    A dataset class for processing and handling Airbnb data collected from https://insideairbnb.com/get-the-data/.
    Processing multiple cities at once is supported.
    Data can either be read in from raw data files or from already (via this script) preprocessed data files.


    Important Attributes:
        raw_data_dict (Dict): If files are read in from raw data directory, raw_data_dict contains them.
        all_cities_listings (pd.DataFrame): Pandas Dataframe containing all listings of respective cities.
    
    """
    def __init__(
            self,
            raw_data_dir: str = "/kaggle/input/berlin-amsterdam/raw_data",
            process_all_cities: bool = True,
            cities_to_process: list   = ["berlin"],
            read_from_raw: bool = True,
            preprocessed_data_dir: str = 'preprocessed_data',
            file_name: str = 'single_city_listing.csv'):
        
        self.process_all_cities = process_all_cities
        self._cities_to_process = cities_to_process
        
        # determine whether data is read in from raw data files or already preprocessed (via this script) data files
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
        
    
    def _read_data_from_files(self) -> Dict:
        logging.info(f"reading in data from {self.raw_data_dir}")
        cities_in_raw_data_dir = os.listdir(self.raw_data_dir)

        if not self.process_all_cities and not set(self._cities_to_process).issubset(cities_in_raw_data_dir):
            raise ValueError("not all requested citys are in directory")
        
        raw_data_dict = {}

        if self.process_all_cities:
            self._cities_to_process = cities_in_raw_data_dir

        self.cities = self._cities_to_process
        
        for city in self._cities_to_process:
            logging.info(f"collecting data for city: {city}")
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

        logging.info(f"collecting data process done")

        return raw_data_dict

    def _integrate_reviews_into_listings(self) -> None:
        logging.info(f"initializing reviews collection process and integration into city listings")
        
        for city in self.cities:
            logging.info(f"current city: {city}")
            city_listings = self.raw_data_dict[city]["listings.csv"]
            city_reviews = self.raw_data_dict[city]["reviews.csv"]       
            #city_calendar = self.raw_data_dict[city]["calendar.csv"] 

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
        
        logging.info("integration of reviews into cites listings done")

    def _aggregate_city_listings_into_one_df(self) -> pd.DataFrame: 
        logging.info("initializing aggregation of regional listings into one dataframe")
        all_cities_listings = []

        for city in self.cities:
            city_listings = self.raw_data_dict[city]["listings.csv"]
            city_listings.insert(0, 'city', city)
            all_cities_listings.append(city_listings)

        all_cities_listings = pd.concat(all_cities_listings, ignore_index=True)
        logging.info("aggregation done")
        return all_cities_listings

    def filter_listings_and_impute_nan(self,
                                       meta_data_columns: list[str] = [
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
                                       nan_columns: list[str] = ['calendar_updated'],
                                       include_only_reviewed: bool = True) -> None:
        
        all_cities_listings = self.all_cities_listings
        #filter the NaN columns (here are no entries)
        all_cities_listings = all_cities_listings.drop(columns = nan_columns)
        
        #filter the meta_data columns (these columns are not used for prediction)
        all_cities_listings = all_cities_listings.drop(columns = meta_data_columns)
    

        #categorical NaN:
        #  'host_response_time' --> extra NaN category
        all_cities_listings.fillna({'host_response_time': 'not available'}, inplace=True)
    
        #  'host_is_superhost'  --> False
        all_cities_listings.fillna({'host_is_superhost': 'f'}, inplace=True)
        
        #  'has_availability'   --> False
        all_cities_listings.fillna({'has_availability': 'f'}, inplace=True)

        
        #numerical NaN:
        #  'host_response_rate'    --> mean/median
        all_cities_listings['host_response_rate'] = all_cities_listings['host_response_rate'].map(
            lambda s: s.removesuffix('%') if type(s) == str else s, 
            na_action = 'ignore'
        )
        all_cities_listings['host_response_rate'] = all_cities_listings['host_response_rate'].str.rstrip('%').astype('float')
        median_value = all_cities_listings['host_response_rate'].dropna().median()
        all_cities_listings['host_response_rate'].fillna(value=median_value, inplace=True)

        #  'host_acceptance_rate'  --> mean/median
        all_cities_listings['host_acceptance_rate'] = all_cities_listings['host_acceptance_rate'].map(
            lambda s: s.removesuffix('%') if type(s) == str else s, 
            na_action = 'ignore'
        )
        all_cities_listings['host_acceptance_rate'] = all_cities_listings['host_acceptance_rate'].str.rstrip('%').astype('float')
        median_value = all_cities_listings['host_acceptance_rate'].dropna().median()
        all_cities_listings['host_acceptance_rate'].fillna(value=median_value, inplace=True)

        #  'bathrooms'        --> mean/median
        median_value = all_cities_listings['bathrooms'].dropna().median()
        all_cities_listings['bathrooms'].fillna(value=median_value, inplace=True)

        #  'bedrooms'         --> mean/median
        median_value = all_cities_listings['bedrooms'].dropna().median()
        all_cities_listings['bedrooms'].fillna(value=median_value, inplace=True)

        #  'beds'         --> mean/median
        median_value = all_cities_listings['beds'].dropna().median()
        all_cities_listings['beds'].fillna(value=median_value, inplace=True)

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
        
        self.all_cities_listings = all_cities_listings
        
    
    def add_nlp_embedding(self, 
                          nlp_col_names: list[str] = ['name', 'description', 'neighborhood_overview', 'host_about', 'amenities','comments'], 
                          batch_size: int = 32) -> None:
        logging.info(f"initializing NLP embedding process, batch size: {batch_size}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_name = 'distilbert-base-multilingual-cased'
        tokenizer = tf.AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        model = tf.AutoModel.from_pretrained(model_name).to(device)
        logging.info(f"embeddings are computed using transformer model: {model_name} from hugging face")
        
        for nlp_col_name in nlp_col_names:
            logging.info(f"current nlp column: {nlp_col_name}")

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
                                 col_names: list[str] = [
                                    'name_emb', 
                                    'description_emb', 
                                    'neighborhood_overview_emb', 
                                    'host_about_emb', 
                                    'amenities_emb',
                                    'comments_emb',
                                    'host_picture_emb',
                                    'picture_emb'
                                 ],
                                keep_variance: float = 0.95) -> None:
        
        logging.info("initializing dimensionality reduction")
            
        for col_name in col_names:
            logging.info(f"current embeddings: {col_name}")
            col = self.all_cities_listings[col_name]
            col_array = np.asarray([np.asarray(entry) for entry in col])

            pca = PCA(n_components = keep_variance, svd_solver='full')
            pca.fit(col_array)
            dim_red_col_array = pca.transform(col_array)
            logging.info(f"used {pca.n_components_ } components for dim reduction to explain {keep_variance*100}% of the data")
            
            dim_red_col_name = col_name + '_dim_red'
            self.all_cities_listings[dim_red_col_name] = list(dim_red_col_array)
        logging.info("dimensionality reduction done")

    def download_images_and_save(self, 
                            image_url_col_names: list[str] = ['host_picture_url','picture_url'], 
                            saving_dir: str = 'kaggle/working/preprocessed_data/filtered_dataset_images',
                            process_n_images: int = -1) -> None:
        
        image_downloader = ImageDownloader(cities=self.cities, all_cities_listings=self.all_cities_listings)
        image_downloader.download_images_and_save(
                            image_url_col_names=image_url_col_names, 
                            saving_dir=saving_dir, 
                            process_n_images=process_n_images)
        
                
            

    def add_image_embedding(self, 
                            image_url_col_names: list[str] = ['host_picture_url','picture_url'], 
                            batch_size: int = 32,
                            read_from_dir: bool = False,
                            read_dir: str = 'kaggle/working/preprocessed_data/filtered_dataset_images',
                            embedd_n_images: int = -1) -> None:
        
        logging.info("initializing image embedding process")
        
        for image_url_col_name in image_url_col_names:
            
            if read_from_dir:
                logging.info(f"reading images from directory for column '{image_url_col_name}'")
                
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
                logging.info(f"downloading images from web for column '{image_url_col_name}'")
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
                            except Exception as e:
                                no_access_indices.append(i)
                                image_list.append(Image.new("RGB", image_size))  
                        else:
                            no_access_indices.append(i)
                            image_list.append(Image.new("RGB", image_size))
                            #response.raise_for_status()
        
                logging.info(f"pictures from rows {no_access_indices} could not be accessed")
            logging.info("transform images and construct dataloader")
    
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
            
            logging.info("embedding image data using ResNet50")
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
            
            for no_access_index in no_access_indices:
                feature_embeddings_list[no_access_index] = mean_embedding
    
            self.all_cities_listings[image_col_embedded_name] = feature_embeddings_list
            
        logging.info("image embedding done")
    
    def save_all_cities_listings_to_file(self, 
                                         file_name: str = "single_city_listing.csv", 
                                         saving_dir: str =  'preprocessed_data',
                                         single_data_frames: bool = True) -> None:
        
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
                logging.info(f"{city} listings saved to path: {file_path}")
        else:
            file_path = saving_dir + '/' + file_name 
            self.all_cities_listings.to_csv(file_path)
            logging.info(f"all cities listings saved to path: {file_path}")

    def _read_preprocessed_listings(self, file_name: str) -> pd.DataFrame:
        
        cities_subdirectories = [d for d in os.listdir(self.preprocessed_data_dir) if os.path.isdir(os.path.join(self.preprocessed_data_dir, d))]

        
        if not set(set(self._cities_to_process)).issubset(cities_subdirectories):
            raise ValueError("not all cities in need to be processed are in given read directory")

        if self.process_all_cities:
            self._cities_to_process = cities_subdirectories
        self.cities = self._cities_to_process
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
        logging.info("reading preprocessed cities from directoy done")
        return all_cities_listings
    
    def local_currency_to_usd_conversion(self):
        """
            Changes local currency to USD for specified cities in all_cities_listings dataframe
        """
        cities = self.cities
        all_cities_listings = self.all_cities_listings

        # every cities currency should be mapped to USD via removing USD sign from string and converting the number to float then multiplying by the exchange rate
        # for 'berlin', "barcelona", "istanbul", "london", "oslo"
        exchange_rates = {
            "berlin": 1.05, # EUR -> USD
            "barcelona": 1.05, # EUR -> USD
            "istanbul": 0.026, # TRY -> USD
            "london": 1.26, # GBP -> USD
            "oslo": 0.09, # NOK -> USD
            "los_angeles": 1.0 # USD -> USD
        }

        # price is stored in all_cities_listings["price"] as string with currency sign
        for city in cities:
            exchange_rate = exchange_rates[city]
            all_cities_listings.loc[all_cities_listings["city"] == city, "price"] = all_cities_listings.loc[all_cities_listings["city"] == city, "price"].apply(lambda x: "$" + str(float(x[1:].replace(",", "")) * exchange_rate))

        self.all_cities_listings = all_cities_listings
        logging.info("local currency to USD conversion done")

    def categorical_to_one_hot_encoding(self,
                                        include_city_column: bool = False) -> None:
        
        binary_columns = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'has_availability', 'instant_bookable']
        categorical_columns = ['host_response_time', 'room_type']

        if include_city_column:
            categorical_columns.append('city')

        all_cities_listings = self.all_cities_listings

        all_cities_listings = pd.get_dummies(all_cities_listings, columns=binary_columns, drop_first=True)
        all_cities_listings = pd.get_dummies(all_cities_listings, columns=categorical_columns, drop_first=False)
        
        # Convert the one-hot encoded columns to integers
        for col in all_cities_listings.columns:
            if all_cities_listings[col].dtype == 'bool':
                all_cities_listings[col] = all_cities_listings[col].astype(int)

        self.all_cities_listings = all_cities_listings
        logging.info("one hot encoding done")
        
            



        

def main() -> None:
    data_set = InsideAirbnbDataset(raw_data_dir= "/media/sn/Frieder_Data/Master_Machine_Learning/data",
            process_all_cities = False,
            cities_to_process = ["preprocessed"],
            read_from_raw = False,
            preprocessed_data_dir = '/media/sn/Frieder_Data/Master_Machine_Learning/data_preprocessed',
            file_name = 'european_cities_data.csv')

    #data_set.local_currency_to_usd_conversion()
    #data_set.filter_listings_and_impute_nan()
    data_set.categorical_to_one_hot_encoding(include_city_column= False)
    #data_set.local_currency_to_usd_conversion()
    

    #data_set.download_images_and_save(
    #                        saving_dir = '/media/sn/Frieder_Data/Master_Machine_Learning/images',
    #                        process_n_images = -1)
    #additional_features = ['host_picture_analysis', 'listing_picture_analysis', 'distance_to_city_center', 'average_review_length', 'review_sentiment', 'spelling_errors', 'aesthetic_score', ]
    #AddCustomFeatures(data = data_set.all_cities_listings, additional_features= additional_features)
    data_set.save_all_cities_listings_to_file(saving_dir='/media/sn/Frieder_Data/Master_Machine_Learning/data_preprocessed', single_data_frames= False)


if __name__ == "__main__":
    main()

    







