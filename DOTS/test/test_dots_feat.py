import pytest
from DOTS.dots_feat import chunk_text, featurize_stories, get_data, process_hit, process_data
from datetime import datetime

def test_get_data(n=10):
    data = get_data(n)
    assert len(data['hits']['hits']) == n

def test_process_hit(data=get_data(1)):
    text, date, loc, title = process_hit(data['hits']['hits'][0])
    assert isinstance(datetime.strptime(date, '%d-%m-%Y'),datetime)

def test_featurize_stories(data=get_data(5)):
    articles = process_data(data)
    assert len(articles) == 5
    try:  #since some stories will be unretreatable
        features = featurize_stories(str(articles), 4, 512)
        assert len(features) == 4
    except:
        pass
