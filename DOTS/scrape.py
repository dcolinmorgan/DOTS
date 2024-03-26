import subprocess, json, argparse, os,requests
from opensearchpy import OpenSearch
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime, timedelta
from gnews import GNews
import xml.etree.ElementTree as ET
load_dotenv()
os_url = os.getenv('OS_TOKEN')
lobstr_key = os.getenv('LOBSTR_KEY')

def get_OS_data(n):
    bash_command = f"""
    curl -X GET "{os_url}/emergency-management-news/_search" -H 'Content-Type: application/json' -d '{{
    "_source": ["metadata.GDELT_DATE", "metadata.page_title","metadata.DocumentIdentifier", "metadata.Organizations","metadata.Persons","metadata.Themes","metadata.text", "metadata.Locations"],
        "size": {n},
        "query": {{
            "bool": {{
                "must": [
                    {{"match_all": {{}}}}
                ]
            }}
        }}
    }}'
    """
    process = subprocess.run(bash_command, shell=True, capture_output=True, text=True)
    output = process.stdout
    data = json.loads(output)
    return data


def get_massive_OS_data(t=1):
    client = OpenSearch(os_url)
    query = {
        "size": "100",
        "timeout": "10s",
        "slice": {
            "id": 0,
            "max": 10
        },
        "query": {
            "bool": {
                "must": [
                    {"match_all": {}},
                ]}
            },
        "_source": ["metadata.GDELT_DATE", "metadata.page_title","metadata.DocumentIdentifier", "metadata.Organizations","metadata.Persons","metadata.Themes","metadata.text", "metadata.Locations"],
    }
    response = client.search(
        scroll=str(t)+'m',
        body=query,
    )

    return response, client

def get_google_news(theme,n=10000):
    google_news = GNews()

    google_news.period = '7d'  # News from last 7 days
    google_news.max_results = n  # number of responses across a keyword
    # google_news.country = 'United States'  # News from a specific country 
    google_news.language = 'english'  # News in a specific language
    google_news.exclude_websites = ['yahoo.com', 'cnn.com']  # Exclude news from specific website i.e Yahoo.com and CNN.com
    # google_news.start_date = (2024, 1, 1) # Search from 1st Jan 2020
    # google_news.end_date = (2024, 3, 1) # Search until 1st March 2020

    json_resp = google_news.get_news(theme)
    article=[]

    for i in tqdm(range(len(json_resp)), desc="grabbing directly from GoogleNews"):
        aa=(google_news.get_full_article(json_resp[i]['url']))
        try:
            date=aa.publish_date.strftime("%d-%m-%Y")
        except:
            date=None
        try:
            title=aa.title
            text=aa.text
        except:
            title=None
            text=None
        article.append([title,date,text])

    return article


def get_npr_news(p):
    # Send a GET request to the NPR API
    r = requests.get("http://api.=1m.org/query?apiKey="+npr_key[0], params=p)

    # Parse the XML response to get the story URLs
    root = ET.fromstring(r.content)
    story_urls = [story.find('link').text for story in root.iter('story')]

    # For each story URL, send a GET request to get the HTML content
    full_stories = []
    for url in story_urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the main content of the story. This will depend on the structure of the webpage.
        # Here, we're assuming that the main content is in a <p> tag. You might need to adjust this depending on the webpage structure.
        story = soup.find_all('p')

        # Extract the text from the story
        full_story = ' '.join(p.text for p in story)
        full_stories.append(full_story)
    return full_stories


# def pull_lobstr(page=1, limit=100):
#     values = {
#         "cluster": "65b9eea6e1cc6bb9f0cd2a47751a186f"
#     }
#     headers = {
#         'Content-Type': 'application/json',
#         'Authorization': str({lobstr_key})
#     }
#     url = f'https://api.lobstr.io/v1/runs?page={page}&limit={limit}'
#     responseA = requests.get(url, headers=headers, params=values)


#     url = f'https://api.lobstr.io/v1/runs/{responseA.json()['run_id']}/download'
#     response = requests.get(url, headers=headers)
    
#     return response.json()
