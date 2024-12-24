import numpy as np
import pandas as pd
import os

PATH_TO_REPO = "C:/Users/nilsk/Dokumente/Machine Learning (MSc.)/1. Semester/Data Literacy/DataLit-InsideAirbnb"
RAW_DATA_DIR = PATH_TO_REPO + '/data/raw_data'
SAVING_DIR = PATH_TO_REPO + '/data/preprocessed_data'
CITY_LIST   = ["berlin"]
FILE_NAMES =  ["listings.csv"] # ["listings.csv", "reviews.csv"] Remark: reviews.csv takes orders of magnitude longer


def convert_to_csv(raw_data, index_list):
    """ 
    Convert the buggy CSV file downloaded from InsideAirbnb to proper CSV file format.

    Parameters: 
    raw_data (list of strings): Contains an ordered list of lines from a  raw CSV file with bad format
    index_list (list of integers): Contains the list of indices the proper CSV file should follow

    Returns: 
    csv_lines (list of strings): The list of lines in the proper CSV file format
    """
    
    #extract header from raw data
    header = raw_data[0]
    raw_data = raw_data[1:]

    csv_lines = [header]
    for line in raw_data:
        first_content_in_line = line.split(',')[0]

        #remove newlines
        line = line.replace("\n", "")

        #tests whether specific line in raw data is start of a new entry
        if first_content_in_line.isdigit() and int(first_content_in_line) in index_list:
            csv_lines.append(line)
        else:
            csv_lines[-1] += line

    return csv_lines



#convert raw csv data for in city_list specified cities
def preprocessing_city_csv_files(city_list):
    """ 
    Converts raw data in RAW_DATA_DIR to proper CSV file format for cities specified in CITY_LIST (see above for global settings).
    Converted files are saved in SAVING_DIR.
    
    """
    cities_in_raw_data_dir = os.listdir(RAW_DATA_DIR)

    if not set(CITY_LIST).issubset(cities_in_raw_data_dir):
        raise ValueError("not all requested citys are in directory")
    
    for city in CITY_LIST:
        city_dir = RAW_DATA_DIR + '/' + city
        data_versions = os.listdir(RAW_DATA_DIR + '/' + city)
        
        for data_version in data_versions:
            data_version_dir = city_dir + '/' + data_version
            

            for file_name in FILE_NAMES:
                file_directory = data_version_dir + '/' + file_name 
                file_summary_directory = data_version_dir + '/summary_information/' + file_name

                print(f"processing file: {file_directory}")

                with open(file_directory, 'r', encoding='utf-8') as file:
                    raw_data = file.readlines()

                file_summary_dataframe = pd.read_csv(file_summary_directory, index_col=0)
                index_list = file_summary_dataframe.index.tolist()

                csv_lines = convert_to_csv(raw_data, index_list)

                # Datei schreiben
                file_saving_path = SAVING_DIR + '/' + city + '/' + data_version + '/' + file_name 
                os.makedirs(os.path.dirname(file_saving_path), exist_ok=True)

                with open(file_saving_path, mode="w", encoding="utf-8") as file:
                    for line in csv_lines:
                        file.write(f"{line}\n")



                










if __name__ == "__main__":
    print("initializing preprocessing")
    preprocessing_city_csv_files(CITY_LIST)
    print("preprocessing done")







