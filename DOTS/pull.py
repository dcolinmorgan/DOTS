import json, logging, requests, csv, concurrent.futures
from tqdm import tqdm
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
from .scrape import scrape_lobstr

def process_hit(hit):
    text = []
    source = hit['_source']
    date = datetime.strptime(source['metadata']['GDELT_DATE'], "%Y%m%d%H%M%S")
    date = date.strftime("%d-%m-%Y")
    loc = source['metadata']['Locations']
    loc = loc.replace("'", '"')  # json requires double quotes for keys and string values
    try:
        list_of_dicts = json.loads(loc)
        location_full_names = [dict['Location FullName'] for dict in list_of_dicts if 'Location FullName' in dict]
        loc = location_full_names[0]
    except:
        loc = None
    org = source['metadata']['Organizations']
    per = source['metadata']['Persons']
    theme = source['metadata']['Themes'].rsplit('_')[-1]
    title = source['metadata']['page_title']
    url = source['metadata']['DocumentIdentifier']
    try:
        response = requests.get(url,timeout=5)
    except requests.exceptions.ConnectionError:  #
        logging.debug(f"timeout for {url}")
        return date,loc,title,org,per,theme,text,url
    if response.status_code != 200:
        logging.debug(f"Failed to get {url}")
        return date,loc,title,org,per,theme,text,url
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all(['p'])
    if not paragraphs:
        logging.debug(f"No <p> tags in {url}")
        return date,loc,title,org,per,theme,text,url
    for p in paragraphs:
        text.append(p.get_text())
    return date,loc,title,org,per,theme,text,url

def process_data(data,fast=1):
    articles = []
    results=[]
    hits = data['hits']['hits']
    if fast==0:
        for hit in tqdm(hits, desc="attempting to grab text from url"):
            try:
                results.append(process_hit(hit))
            except:
                logging.debug(f"Grabbing the url stalled after 5s, skipping...")
                pass
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_hit_with_timeout, hits), total=len(hits), desc="attempting to grab text from url"))
    for date,loc,title,org,per,theme,text,url in results:
        if loc is None:
            logging.debug(f"No location info, grabbing from org...")
            loc = org
        if text == None or text == []:
            logging.debug(f"No text from url available, using org/persons/theme instead...")
            articles.append([None,date,loc,title,org,per,theme])
            # writer.writerow([None,date,loc,title,org,per,theme])
            # writer.writerow(['\n'])
        else:
            articles.append([date,loc,title,org,per,theme,text,url])
            # writer.writerow([date,loc,title,org,per,theme,text,url])
            # writer.writerow(['\n'])
    return articles


def process_response(response):
    hits = response["hits"]["hits"]
    output=[]
    text=[]
    # for hit in hits:
    source = hit["_source"]
    # print(source)
    try:
        date = datetime.strptime(source['metadata']['GDELT_DATE'], "%Y%m%d%H%M%S")
        date = formatted_date = date.strftime("%d-%m-%Y")
        loc = source['metadata']['Locations']
        loc = loc.replace("'", '"')  # json requires double quotes for keys and string values
        try:
            list_of_dicts = json.loads(loc)
            location_full_names = [dict['Location FullName'] for dict in list_of_dicts if 'Location FullName' in dict]
            loc = location_full_names[0]
        except:
            loc = None
        org = source['metadata']['Organizations']
        per = source['metadata']['Persons']
        theme = source['metadata']['Themes'].rsplit('_')[-1]
        title = source['metadata']['page_title']
        url = source['metadata']['DocumentIdentifier']
        # output.append([date, loc, title, org, per, theme, url])
        try:
            response = requests.get(url,timeout=5)
        except requests.exceptions.ConnectionError:  #
            logging.debug(f"timeout for {url}")
            return text,date,loc,title,org, per, theme
        if response.status_code != 200:
            logging.debug(f"Failed to get {url}")
            return date,loc,title,org,per,theme,text,url
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all(['p'])
        if not paragraphs:
            logging.debug(f"No <p> tags in {url}")
            return date,loc,title,org,per,theme,text,url
        for p in paragraphs:
            text.append(p.get_text())
            return date,loc,title,org,per,theme,text,url
        # output.append([date, loc, title, org, per, theme, url])
    except:
        pass


def extract_location(location_str):
    if location_str:
        try:
            location_list = json.loads(location_str.replace("'", '"'))
            return [dict['Location FullName'] for dict in location_list if 'Location FullName' in dict]
        except json.JSONDecodeError:
            return None
    else:
        return None


def process_url(url):
    try:
        response = requests.get(url,timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all(['p'])
        text = []
        for p in paragraphs:
            text.append(p.get_text())
        return text
    except Exception as e:
        # print(f"Error processing URL {url}: {e}")
        return None


def pull_data(articles):
    # aa = [item for sublist in articles for item in sublist]
    data = [list(d['_source']['metadata'].values()) for d in articles]
    try:
        df = pd.DataFrame(data, columns=['date','title', 'person', 'org', 'location', 'theme', 'text', 'url'])
        df.date=pd.to_datetime(df.date).dt.strftime('%d-%m-%Y')
        df['locc'] = df['location'].apply(extract_location)
    except:
        df = pd.DataFrame(data, columns=['url','title'])
    with concurrent.futures.ThreadPoolExecutor() as executor:
        df['text'] = list(tqdm(executor.map(process_url, df['url']), total=len(df['url']),desc="grabbing text from url"))
    return df['text'].values.tolist()


def pull_lobstr_gdoc(pull=1):
    articles = pd.read_parquet('DOTS/input/lobstr_text.parquet')
    if pull==0:
        url = 'https://docs.google.com/spreadsheets/d/178sqEWzqubH0znhx7Z6u9ig2EjCRvl0dUsA7b6hQpmY/export?format=csv'
        df = pd.read_csv(url)
    else:
        df = scrape_lobstr()
    # if the story text is already gathered, process and return in list of lists format
    if len(articles) == len(df):  
        logging.info("Using cached lobstr data")
        try:
            df = articles.dropna()
            df = df[df['text'].apply(lambda x: (x) !="[]")]
            df.reset_index(inplace=True)
        except:
            df = articles

    # otherwise gather the story text from the URLs and save to parquet so that subsequent runs dont need to request again
    else:
        df = df[['published_at','url','title','short_description']]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            df['text'] = list(tqdm(executor.map(process_url, df['url']), total=len(df['url'])))
        df = df[['published_at','short_description','text']]
        df.to_parquet('DOTS/input/lobstr_text.parquet')
    return df.values.tolist()
