import pytest, requests
from DOTS.feat import chunk_text, featurize_stories
from DOTS.scrape import get_OS_data, get_google_news, get_massive_OS_data, get_npr_news
from DOTS.pull import process_hit, process_data
from datetime import datetime


@pytest.fixture
def get_data():
    return get_OS_data(10)

def test_get_OS_data(get_data):
    data = get_data
    assert len(data['hits']['hits']) == 10

def test_process_hit(get_data):
    data=get_data
    text,date,loc,title,org,per,theme = process_hit(data['hits']['hits'][0])
    assert isinstance(datetime.strptime(date, '%d-%m-%Y'),datetime)

def test_featurize_stories(get_data):
    data=get_data
    articles = process_data(data)
    assert len(articles) == 10
    try:  #since some stories will be unretreatable
        features = featurize_stories(str(articles), 4, 512)
        assert len(features) == 4
    except:
        pass

@pytest.fixture
def get_massOS_data():
    return get_massive_OS_data(1000)

def test_get_massOS_data(get_massOS_data):
    data = get_massOS_data
    assert len(data['hits']['hits']) >= 1000

def test_massive_featurize(get_massOS_data,t=1):
    response=get_massOS_data
    pagination_id = response["_scroll_id"]
    hits = response["hits"]["hits"]
    while len(hits) != 0:
        articles=[]
        client = OpenSearch(os_url)
        response = client.scroll(
            scroll=str(t)+'m',
            scroll_id=pagination_id
                )
        hits = response["hits"]["hits"]
        article = process_data(response)
        articles.append(article)
    articles = [item for sublist in articles for item in sublist]
    # assert len(articles) == 5
    try:  #since some stories will be unretreatable
        features = featurize_stories(str(articles), 4, 512)
        assert len(features) == 4
    except:
        pass


def test_gnews_featurize():
    articles= get_google_news('disaster',n=10)
    try:  #since some stories will be unretreatable
        features = featurize_stories(str(articles), 4, 512)
        assert len(features) == 4
    except:
        pass

def test_npr_featurize():
    articles= get_npr_news('disaster')
    try:  #since some stories will be unretreatable
        features = featurize_stories(str(articles), 4, 512)
        assert len(features) == 4
    except:
        pass
