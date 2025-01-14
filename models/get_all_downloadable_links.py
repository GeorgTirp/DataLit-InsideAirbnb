import requests
from bs4 import BeautifulSoup
import os
import requests


url = "https://insideairbnb.com/get-the-data/"

# Fetch the page content
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all links that match the dataset format
links = soup.find_all('a', href=True)

# Filter links that contain 'data'
dataset_links = [link['href'] for link in links if 'data' in link['href']]

# Further filter links that contain 'data'
dataset_links_csv = [link for link in dataset_links if 'csv' in link]
dataset_links_geojson = [link for link in dataset_links if 'geojson' in link]

# Print out the links
for link in dataset_links_csv:
    print(f"{link}")


for link in dataset_links_geojson:
    link_list = link.split('/')

    country = link.split('/')[3]
    if len(link_list) > 6:
        print('Its a city')
        city = link.split('/')[5]
        # Create a folder for the country if it doesn't exist
        country_folder = os.path.join('/media/sn/Frieder_Data/Master_Machine_Learning/DataLit-InsideAirbnb/data', country)
        if not os.path.exists(country_folder):
            os.makedirs(country_folder)
        if not os.path.exists(os.path.join(country_folder, city)):
            os.makedirs(os.path.join(country_folder, city))
        
        # Download the file
        response = requests.get(link)

        # Check if the request was successful
        if response.status_code == 200:
            # Save the file in the specified folder
            with open(os.path.join(os.path.join(country_folder, city), link_list[-1]), "wb") as file:
                file.write(response.content)
            print(f"Downloaded and saved {link_list[-1]} to {os.path.join(country_folder, city)}")
        else:
            print(f"Failed to download {link_list[-1]} from {link}")

    else:
        print('Is the whole country')
        # Create a folder for the country if it doesn't exist
        country_folder = os.path.join('/media/sn/Frieder_Data/Master_Machine_Learning/DataLit-InsideAirbnb/data', country)
        if not os.path.exists(country_folder):
            os.makedirs(country_folder)
        
        # Download the file
        response = requests.get(link)
        # Check if the request was successful
        if response.status_code == 200:
            # Save the file in the specified folder
            with open(os.path.join(os.path.join(country_folder), link_list[-1]), "wb") as file:
                file.write(response.content)
            print(f"Downloaded and saved {link_list[-1]} to {os.path.join(country_folder)}")
        else:
            print(f"Failed to download {link_list[-1]} from {link}")


