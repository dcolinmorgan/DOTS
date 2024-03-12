from bs4 import BeautifulSoup
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModel, AutoTokenizer
import torch, subprocess, json, requests, spacy,nltk, string, csv
nltk.download('punkt')  # run once

model_name = "distilroberta-base"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# !python -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm')
n_gram_range = (1, 2)
stop_words = "english"
embeddings=[]
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
