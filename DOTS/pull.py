import json, logging, requests, csv, concurrent.futures, signal
from tqdm import tqdm
import concurrent.futures, requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd

def handler(signum, frame):
    raise TimeoutError()
signal.signal(signal.SIGALRM, handler)

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
        response = requests.get(url)
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


def process_hit_with_timeout(hit):
    try:
        signal.alarm(5)
        return process_hit(hit)
    except:
        logging.debug(f"Grabbing the url stalled after 5s, skipping...")
        return None

def process_data(data,fast=1):
    articles = []
    results=[]
    hits = data['hits']['hits']
    if fast==0:
        for hit in tqdm(hits, desc="attempting to grab text from url"):
            try:
                results.append(process_hit(hit))
                signal.alarm(5)
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
    signal.alarm(0)
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
            response = requests.get(url)
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
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all(['p'])
    text = []
    for p in paragraphs:
        text.append(p.get_text())
    return text


def pull_data(articles):
    # aa = [item for sublist in articles for item in sublist]
    data = [list(d['_source']['metadata'].values()) for d in articles]
    df = pd.DataFrame(data, columns=['date','title', 'person', 'org', 'location', 'theme', 'text', 'url'])
    df.date=pd.to_datetime(df.date).dt.strftime('%d-%m-%Y')
    df['locc'] = df['location'].apply(extract_location)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        df['text'] = list(executor.map(process_url, df['url']))
    return df.values.tolist()
