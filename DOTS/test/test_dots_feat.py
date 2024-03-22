import pytest, requests
from DOTS.feat import chunk_text, featurize_stories
from DOTS.scrape import get_OS_data, get_google_news, get_massive_OS_data, get_npr_news
from DOTS.pull import process_hit, process_data, pull_data
from datetime import datetime


@pytest.fixture
def get_data():
    return get_OS_data(10)

def test_get_OS_data(get_data):
    data = get_data
    assert len(data['hits']['hits']) == 10

def test_process_hit(get_data):
    data=get_data
    date,loc,title,org,per,theme,text,url = process_hit(data['hits']['hits'][0])
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
    return get_massive_OS_data(100)

def test_get_massOS_data(get_massOS_data):
    data, ext = get_massOS_data
    assert len(data['hits']['hits']) == 100
    
@pytest.fixture
def get_scroll_data():
        response, client = get_massive_OS_data(1)
        pagination_id = response["_scroll_id"]
        hits = response["hits"]["hits"]
        articles=[]
        articles2=[]
        while len(hits) != 0 and len(articles2) < 11000:
            response = client.scroll(
                scroll='1m',
                scroll_id=pagination_id
                    )
            hits = response["hits"]["hits"]
            # article = process_data(response)
            articles.append(hits)
            articles2 = [item for sublist in articles for item in sublist]
        return [item for sublist in articles for item in sublist]

def test_scroll_data(get_scroll_data):
    articles = get_scroll_data
    assert len(articles) == 11000

def test_massive_featurize(get_scroll_data,t=1):
    articles = get_scroll_data
    stories=pull_data(articles[:100]) ## test full get but partial pull
    try:
        features = featurize_stories(str(stories), 4, 512)
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

# def test_npr_featurize():
#     articles= get_npr_news('disaster')
#     try:  #since some stories will be unretreatable
#         features = featurize_stories(str(articles), 4, 512)
#         assert len(features) == 4
#     except:
#         pass
