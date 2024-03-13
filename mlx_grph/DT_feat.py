# conda create -n DT ipykernel 
# python -m ipykernel install --user --name DT
# pip install torch bs4 transformers spacy numpy pandas scikit-learn scipy nltk
from bs4 import BeautifulSoup
import argparse
import numpy as np
import concurrent.futures
# from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModel, AutoTokenizer
import torch, spacy,nltk,subprocess, json, requests,string,csv
# nltk.download('punkt')  # run once

parser = argparse.ArgumentParser(description='Process OS data for dynamic features.')
parser.add_argument('-n', type=int, default=10, help='Number of data items to get')
parser.add_argument('-f', type=int, default=13, help='Number of features per item to get')
parser.add_argument('-o', '--output', default='OS_feats.csv', help='Output file name')
args = parser.parse_args()


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


def chunk_text(text, max_len):
    # Tokenize the text into tokensconda
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



def featurize_stories(text, top_k, max_len):
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
    # candidate_tokens = {k: v.to(device) for k, v in (candidate_tokens).items()}
    candidate_embeddings = model(**candidate_tokens)["pooler_output"]
    candidate_embeddings = candidate_embeddings.detach()#.to_numpy()

    # words = nltk.word_tokenize(text)
    # chunks = [words[i:i + 512] for i in range(0, len(words), 512)]
    chunks = chunk_text(text, max_len)  # use this to chunk better and use less padding thus less memory but also less affect from averging

    for chunk in chunks:
        text_tokens = tokenizer(chunk, padding=True, return_tensors="pt")
        # text_tokens = {k: v.to(device) for k, v in (text_tokens).items()}
        text_embedding = model(**text_tokens)["pooler_output"]
        text_embedding = text_embedding.detach()#.to_numpy()
        embeddings.append(text_embedding)
    max_emb_shape = max(embedding.shape[0] for embedding in embeddings)
    padded_embeddings = [np.pad(embedding.cpu(), ((0, max_emb_shape - embedding.shape[0]), (0, 0))) for embedding in embeddings]
    avg_embedding = np.min(padded_embeddings, axis=0)
    distances = cosine_similarity(avg_embedding, candidate_embeddings.cpu())
    torch.cuda.empty_cache()
    return [candidates[index] for index in distances.argsort()[0][::-1][-top_k:]]


def get_data(n=args.n):
    bash_command = f"""
    curl -X GET "https://louie_armstrong:peach-Jam-42-prt@search-opensearch-dev-domain-7grknmmmm7nikv5vwklw7r4pqq.us-east-1.es.amazonaws.com/emergency-management-news/_search" -H 'Content-Type: application/json' -d '{{
        "_source": ["metadata.GDELT_DATE", "metadata.page_title", "metadata.DocumentIdentifier"],
        "size": {n},
        "query": {{
        "match_all": {{}}
        }}
    }}'
    """
    # print(f"Running command: {bash_command}")  # Print the command that will be run

    process = subprocess.run(bash_command, shell=True, capture_output=True, text=True)

    # print(f"Command output: {process.stdout}")  # Print the output of the command
    # print(f"Command error: {process.stderr}")  # Print the error output of the command

    output = process.stdout
    data = json.loads(output)

    # print(f"Data: {data}")  # Print the data that will be returned

    return data

def get_big_data():
    from elasticsearch import Elasticsearch

    es = Elasticsearch()

    # Initialize the scroll
    page = es.search(
    index = 'your_index',
    scroll = '2m',  # keep the search context alive for 2 minute
    size = 1000,  # number of results per "page"
    body = {
        # your query goes here
    }
    )

    sid = page['_scroll_id']
    scroll_size = len(page['hits']['hits'])

    # Start scrolling
    while scroll_size > 0:
        print("Scrolling...")
        page = es.scroll(scroll_id = sid, scroll = '2m')
        # Update the scroll ID
        sid = page['_scroll_id']
        # Get the number of results that returned in the last scroll
        scroll_size = len(page['hits']['hits'])
        print("scroll size: " + str(scroll_size))
        # Do something with the obtained page

def process_data(data):
    articles=[]
    # Extract the specific fields
    hits = data['hits']['hits']
    with open('output.csv', 'w') as file:
        writer = csv.writer(file)
        for hit in hits:
            z=[]
            source = hit['_source']
            GDELT_DATE = source['metadata']['GDELT_DATE']
            # Locations = source['metadata']['Locations']
            # Extras = source['metadata']['Extras']
            page_title = source['metadata']['page_title']
            url = source['metadata']['DocumentIdentifier']
            
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            # print(soup.find('content'))
            for p in soup.find_all('p'):
                z.append(p.get_text())
            writer.writerow(z)
            writer.writerow(['\n'])
            articles.append(z)

    with open('output.html', 'w') as file:
        file.write(str(soup))
        
    with open('output.csv', 'w') as file:
        file.write(str(z))
    
    return articles


def process_hit(hit):
    z = []
    source = hit['_source']
    GDELT_DATE = source['metadata']['GDELT_DATE']
    page_title = source['metadata']['page_title']
    url = source['metadata']['DocumentIdentifier']

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    for p in soup.find_all('p'):
        z.append(p.get_text())
    return z, soup

def process_data_fast(data):
    articles = []
    hits = data['hits']['hits']

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_hit, hits))

    with open('output.csv', 'w') as file:
        writer = csv.writer(file)
        for z, soup in results:
            writer.writerow(z)
            writer.writerow(['\n'])
            articles.append(z)

    with open('output.html', 'w') as file:
        file.write(str(results[-1][1]))  # Write the soup of the last hit

    # with open('output.csv', 'w') as file:
    #     file.write(str(results[-1][0]))  # Write the z of the last hit

    return articles


data = get_data(args.n)
# data = get_big_data()
# articles = process_data(data)
articles = process_data_fast(data)

rank_articles=[]
for i in articles[1:]:
    try:
        cc=featurize_stories(str(i), top_k = args.f, max_len=512)
        rank_articles.append(cc)
    except IndexError:
        pass

with open('OS_feats.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rank_articles)
