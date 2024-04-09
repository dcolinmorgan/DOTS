import requests, re
from bs4 import BeautifulSoup

def get_species(url='https://funcoup.org/downloads/'):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a')
    text = soup.get_text()
    gz_links = [link for link in links if link['href'].endswith('.gz')]
    sizes = re.findall(r'(\d+\.\d+|\d+)(?=\s*(MB|KB))', text)
    
    species=[]
    for link, size in zip(gz_links, sizes):
        file_name = link.text
        file_size = size
        species.append([file_name, file_size])
    species=pd.DataFrame(species, columns=['file_name', 'file_size'])
    
    def convert_to_kb(size):
        value, unit = size
        value = float(value)
        if unit == 'MB':
            value *= 1024  # convert MB to KB
        return value
    species['type'] = species['file_name'].apply(lambda x: 'full' if 'full' in x else ('compact' if 'compact' in x else 'unknown'))
    species['file_size_kb'] = species['file_size'].apply(convert_to_kb)
    species = species.sort_values(by='file_size_kb', ascending=True)
    species = species.drop(columns=['file_size_kb'])
    species['name']=species['file_name'].str.split('_').str[1]
    name_dict = species['name'].to_dict()
    name_dict = {v: v for k, v in name_dict.items()}
    return name_dict
        
