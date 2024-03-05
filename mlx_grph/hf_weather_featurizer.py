## adapted from https://jaketae.github.io/study/keyword-extraction/#candidate-selection
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModel, AutoTokenizer
from datasets import Dataset
model_name = "distilroberta-base"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
nlp = spacy.load('en_core_web_sm')
n_gram_range = (1, 2)
stop_words = "english"
embeddings=[]

import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
p = {'searchTerm':'"natural disaster"','numResults':'10'}


def get_npr_stories(p):
    # Send a GET request to the NPR API
    r = requests.get("http://api.npr.org/query?apiKey=MDE5Mzg3Mjc2MDE0MzMyMjM3NjM5ZTI2Ng001", params=p)

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

def chunk_text(text, max_len):
    # Tokenize the text into tokens
    tokens = nltk.word_tokenize(text)

    # Calculate the number of chunks and the size of the final chunk
    num_chunks = len(tokens) // max_len
    final_chunk_size = len(tokens) % max_len

    # If the final chunk is too small, distribute its tokens among the other chunks
    if final_chunk_size < max_len / 2:
        num_chunks += 1
        chunk_sizes = [len(tokens) // num_chunks + (1 if i < len(tokens) % num_chunks else 0) for i in range(num_chunks)]
        chunks = [tokens[sum(chunk_sizes[:i]):sum(chunk_sizes[:i+1])] for i in range(num_chunks)]
    else:
        chunks = [tokens[i:i + max_len] for i in range(0, len(tokens), max_len)]

    return chunks


def featurize_stories(text, max_len, top_k):
    # Extract candidate words/phrases
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([text])
    all_candidates = count.get_feature_names_out()
    doc = nlp(text)
    noun_phrases = set(chunk.text.strip().lower() for chunk in doc.noun_chunks)
    nouns = set()
    for token in doc:
        if token.pos_ == "NOUN":
            nouns.add(token.text)

    all_nouns = nouns.union(noun_phrases)
    candidates = list(filter(lambda candidate: candidate in all_nouns, all_candidates))
    candidate_tokens = tokenizer(candidates, padding=True, return_tensors="pt")
    candidate_embeddings = model(**candidate_tokens)["pooler_output"]
    candidate_embeddings = candidate_embeddings.detach().numpy()

    # words = nltk.word_tokenize(text)
    # chunks = [words[i:i + 512] for i in range(0, len(words), 512)]
    chunks = chunk_text(text, max_len)  # use this to chunk better and use less padding thus less memory but also less affect from averging

    for chunk in chunks:
        text_tokens = tokenizer(chunk, padding=True, return_tensors="pt")
        text_embedding = model(**text_tokens)["pooler_output"]
        text_embedding = text_embedding.detach().numpy()
        embeddings.append(text_embedding)
    max_emb_shape = max(embedding.shape[0] for embedding in embeddings)
    padded_embeddings = [np.pad(embedding, ((0, max_emb_shape - embedding.shape[0]), (0, 0))) for embedding in embeddings]
    avg_embedding = np.min(padded_embeddings, axis=0)
    distances = cosine_similarity(avg_embedding, candidate_embeddings)

    return [candidates[index] for index in distances.argsort()[0][::-1][-top_k:]]


import nltk, string, numpy as np
# nltk.download('punkt')
full_stories = []
for i in ['"extreme-weather"','"natural-disaster"','"epidemic"','"shooting"']:
    p = {'searchTerm':i,'numResults':'50'}
    fs=(get_npr_stories(p))
    full_stories.append(fs)
full_stories = [item for sublist in full_stories for item in sublist]
len(full_stories)
    
for i in range(len(full_stories)):
    cc=featurize_stories(full_stories[i], max_len=512, top_k=4)
    # print(cc)
    rank_articles.append(cc)

import pandas as pd
full_dataset=pd.DataFrame()
full_dataset['story'] = pd.DataFrame(full_stories)
full_dataset[['featA','featB','featC','featD']]=pd.DataFrame(rank_articles)
Dataset.from_pandas(full_dataset).push_to_hub('Dcolinmorgan/extreme-weather-news',token='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')
