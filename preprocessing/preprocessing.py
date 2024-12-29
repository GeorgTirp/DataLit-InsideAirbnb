import numpy as np
import pandas as pd
import os

PATH_TO_REPO = "C:/Users/nilsk/Dokumente/Machine Learning (MSc.)/1. Semester/Data Literacy/DataLit-InsideAirbnb"
RAW_DATA_DIR = PATH_TO_REPO + '/data/raw_data'
SAVING_DIR = PATH_TO_REPO + '/data/preprocessed_data'
PROCESS_ALL_CITIES = True
CITY_LIST   = ["berlin"] #list cities which should be processed if not PROCESS_ALL_CITIES
DEBUG_MODE = False # determines if preprocessing is in DEBUG_MODE (no processing of file --> execution of main-function)


#convert raw csv data for in city_list specified cities
def collecting_data(city_list):
    """ 
    Converts raw data in RAW_DATA_DIR to proper CSV file format for cities specified in CITY_LIST (see above for global settings).
    Converted files are saved in SAVING_DIR.
    
    """
    print("initializing preprocessing")
    cities_in_raw_data_dir = os.listdir(RAW_DATA_DIR)

    if not PROCESS_ALL_CITIES and not set(CITY_LIST).issubset(cities_in_raw_data_dir):
        raise ValueError("not all requested citys are in directory")
    
    data_dict = {}

    if PROCESS_ALL_CITIES:
        CITY_LIST = cities_in_raw_data_dir
    
    for city in CITY_LIST:
        data_dict[city] = {}
        city_dir = RAW_DATA_DIR + '/' + city
        FILE_NAMES = [f for f in os.listdir(city_dir) if os.path.isfile(os.path.join(city_dir, f))]

        for file_name in FILE_NAMES:
            if file_name.endswith('.csv') or file_name.endswith('.geojson') or file_name.endswith('.csv.gz'):
                file_path = os.path.join(city_dir, file_name)
        
                # Read the file into a DataFrame
                if file_name.endswith('.geojson'):
                    df = pd.read_json(file_path)  # Adjust based on the specific geojson handling
                else:
                    df = pd.read_csv(file_path, index_col=0)

                #basename = file_name.split(sep=".")[0]
                
                data_dict[city][file_name] = df
    
    return data_dict

                
    



def main():
    pass


if __name__ == "__main__":
    if not DEBUG_MODE:
        data_dict = collecting_data(CITY_LIST)
        print(f"collected data from {RAW_DATA_DIR} and stored in data dictionary")
    else:
        main()

    









