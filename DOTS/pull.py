import json, logging, requests, csv, concurrent.futures, signal
from tqdm import tqdm
from bs4 import BeautifulSoup
from datetime import datetime

def handler(signum, frame):
    raise TimeoutError()
signal.signal(signal.SIGALRM, handler)

def process_hit(hit):
    text = []
    source = hit['_source']
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
    try:
        response = requests.get(url)
    except requests.exceptions.ConnectionError:  #
        logging.debug(f"timeout for {url}")
        return text,date,loc,title,org,per,theme
    if response.status_code != 200:
        logging.debug(f"Failed to get {url}")
        return text,date,loc,title,org,per,theme
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all(['p'])
    if not paragraphs:
        logging.debug(f"No <p> tags in {url}")
        return text,date,loc,title,org,per,theme
    for p in paragraphs:
        text.append(p.get_text())
    return text,date,loc,title,org,per,theme


def process_data(data,fast=1):
    articles = []
    results=[]
    hits = data['hits']['hits']
    
    for hit in tqdm(hits, desc="attempting to grab text from url"):
        if fast==0:
            try:
                results.append(process_hit(hit))
                signal.alarm(5)
            except:
                logging.debug(f"Grabbing the url stalled after 5s, skipping...")
                pass
        else:
            e = concurrent.futures.ThreadPoolExecutor()
            try:
                future = e.submit(process_hit, hit)
                result = future.result(timeout=5)  # Set timeout for 5 seconds
                results.append(result)
            except concurrent.futures.TimeoutError:
                logging.debug(f"Grabbing the url stalled after 5s, skipping...")

    with open('DOTS/input/feat_input.csv', 'w') as file:
        writer = csv.writer(file)
        for text,date,loc,title,org,per,theme in results:
            if loc is None:
                logging.debug(f"No location info, grabbing from org...")
                loc = org
            if text == None or text == []:
                logging.debug(f"No text from url available, using org/persons/theme instead...")
                articles.append([None,date,loc,title,org,per,theme])
                writer.writerow([None,date,loc,title,org,per,theme])
                writer.writerow(['\n'])
            else:
                articles.append([text,date,loc,title,org,per,theme])
                writer.writerow([text,date,loc,title,org,per,theme])
                writer.writerow(['\n'])
    signal.alarm(0)
    return articles


# def process_response(response):
#     hits = response["hits"]["hits"]
#     output=[]
#     text=[]
#     # for hit in hits:
#     source = hit["_source"]
#     # print(source)
#     try:
#         date = datetime.strptime(source['metadata']['GDELT_DATE'], "%Y%m%d%H%M%S")
#         date = formatted_date = date.strftime("%d-%m-%Y")
#         loc = source['metadata']['Locations']
#         loc = loc.replace("'", '"')  # json requires double quotes for keys and string values
#         try:
#             list_of_dicts = json.loads(loc)
#             location_full_names = [dict['Location FullName'] for dict in list_of_dicts if 'Location FullName' in dict]
#             loc = location_full_names[0]
#         except:
#             loc = None
#         org = source['metadata']['Organizations']
#         per = source['metadata']['Persons']
#         theme = source['metadata']['Themes'].rsplit('_')[-1]
#         title = source['metadata']['page_title']
#         url = source['metadata']['DocumentIdentifier']
#         # output.append([date, loc, title, org, per, theme, url])
#         try:
#             response = requests.get(url)
#         except requests.exceptions.ConnectionError:  #
#             logging.debug(f"timeout for {url}")
#             return text,date,loc,title,org, per, theme
#         if response.status_code != 200:
#             logging.debug(f"Failed to get {url}")
#             return text,date,loc,title,org,per,theme
#         soup = BeautifulSoup(response.text, 'html.parser')
#         paragraphs = soup.find_all(['p'])
#         if not paragraphs:
#             logging.debug(f"No <p> tags in {url}")
#             return text,date,loc,title,org,per,theme
#         for p in paragraphs:
#             text.append(p.get_text())
#             return text,date,loc,title,org,per,theme
#         # output.append([date, loc, title, org, per, theme, url])
#     except:
#         pass
