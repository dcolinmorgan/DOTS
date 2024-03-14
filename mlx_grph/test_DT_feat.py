import pytest
from unittest.mock import MagicMock
from mlx_grph import DT_feat

def test_chunk_text():
    text = "This is a test sentence for chunking text into smaller parts."
    chunks = DT_feat.chunk_text(text, 5)
    assert len(chunks) == 2

def test_featurize_stories(mocker):
    text = "This is a test sentence for featurizing stories."
    mocker.patch('DT_feat.CountVectorizer')
    mocker.patch('DT_feat.tokenizer')
    mocker.patch('DT_feat.model')
    features = DT_feat.featurize_stories(text, 3, 512)
    assert len(features) == 3

def test_get_data(mocker):
    mocker.patch('DT_feat.subprocess.run')
    DT_feat.subprocess.run.return_value = MagicMock(stdout='{"hits": {"hits": []}}')
    data = DT_feat.get_data(10)
    assert data == {"hits": {"hits": []}}

def test_process_hit(mocker):
    hit = {'_source': {'metadata': {'GDELT_DATE': '20220101000000', 'Locations': '[]', 'page_title': 'Test', 'DocumentIdentifier': 'http://test.com'}}}
    mocker.patch('DT_feat.requests.get')
    DT_feat.requests.get.return_value = MagicMock(status_code=200, text='<html><body><p>Test</p></body></html>')
    text, date, loc, title = DT_feat.process_hit(hit)
    assert date == '01-01-2022'
    assert loc == None
    assert title == 'Test'
    assert text == ['Test']

def test_process_data(mocker):
    data = {'hits': {'hits': [{'_source': {'metadata': {'GDELT_DATE': '20220101000000', 'Locations': '[]', 'page_title': 'Test', 'DocumentIdentifier': 'http://test.com'}}}]}}
    mocker.patch('DT_feat.process_hit')
    DT_feat.process_hit.return_value = (['Test'], '01-01-2022', None, 'Test')
    articles = DT_feat.process_data(data)
    assert articles == [[None, '01-01-2022', ['Test']]]
