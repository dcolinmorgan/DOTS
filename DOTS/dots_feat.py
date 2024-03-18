# conda create -n DT ipykernel 
# python -m ipykernel install --user --name DT
# pip install torch bs4 transformers spacy numpy pandas scikit-learn scipy nltk
from bs4 import BeautifulSoup
import argparse, signal
from tqdm import tqdm
from datetime import datetime
import numpy as np
import concurrent.futures
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModel, AutoTokenizer
import torch, spacy,nltk,subprocess, json, requests,string,csv,logging,os
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
load_dotenv()
os_url = os.getenv('OS_TOKEN')

# Setup logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup argument parser
parser = argparse.ArgumentParser(description='Process OS data for dynamic features.')
parser.add_argument('-n', type=int, default=10, help='Number of data items to get')
parser.add_argument('-f', type=int, default=3, help='Number of features per item to get')
parser.add_argument('-o', type=str, default='dots_feats.csv', help='Output file name')
parser.add_argument('-p', type=int, default=1, help='Parallelize requests')
parser.add_argument('-s', type=datetime, default=20230101, help='start date')
parser.add_argument('-e', type=datetime, default=20231231, help='end date')
args, unknown = parser.parse_known_args()

# Load models and tokenizers
model_name = "distilroberta-base"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# !python -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm')

# Define constants
n_gram_range = (1, 2)
stop_words = "english"
embeddings=[]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def handler(signum, frame):
    raise TimeoutError()
signal.signal(signal.SIGALRM, handler)

# Define functions
def get_data(n=args.n, s=args.s, e=args.e):
    bash_command = f"""
    curl -X GET "{os_url}" -H 'Content-Type: application/json' -d '{{
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
        return text,date,loc,title
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


def process_data(data,fast=args.p):
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
                articles.append([loc,date,None,org,per,theme])
                writer.writerow([loc,date,None,org,per,theme])
                writer.writerow(['\n'])
            else:
                articles.append([loc,date,text,None,None,None])
                writer.writerow([loc,date,text,None,None,None])
                writer.writerow(['\n'])
    signal.alarm(0)
    return articles


def chunk_text(text, max_len):
    tokens = nltk.word_tokenize(text)
    num_chunks = len(tokens) // max_len
    final_chunk_size = len(tokens) % max_len
    logging.info(f"chunking artcile into {num_chunks} chunks")
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
    datetimes = set()
    for token in doc:
        if token.pos_ == "NOUN":
            nouns.add(token.text)
        if token.ent_type_ == "DATE":
            datetimes.add(token.text)  # no need to tokenize dates, just add to feature list artificially by repeating
    all_nouns = nouns.union(noun_phrases)
    candidates = list(filter(lambda candidate: candidate in all_nouns, all_candidates))
    candidate_tokens = tokenizer(candidates, padding=True, return_tensors="pt")
    # if device == 'cuda':
    candidate_tokens = {k: v.to(device) for k, v in (candidate_tokens).items()}
    candidate_embeddings = model(**candidate_tokens)["pooler_output"]
    candidate_embeddings = candidate_embeddings.detach()  # .to_numpy
    chunks = chunk_text(text, max_len)  # use this to chunk better and use less padding thus less memory but also less affect from averging
    for chunk in chunks:
        text_tokens = tokenizer(chunk, padding=True, return_tensors="pt")
        # if device == 'cuda':
        text_tokens = {k: v.to(device) for k, v in (text_tokens).items()}
        text_embedding = model(**text_tokens)["pooler_output"]
        text_embedding = text_embedding.detach()#.to_numpy()
        embeddings.append(text_embedding)
    max_emb_shape = max(embedding.shape[0] for embedding in embeddings)
    padded_embeddings = [np.pad(embedding.cpu(), ((0, max_emb_shape - embedding.shape[0]), (0, 0))) for embedding in embeddings]
    avg_embedding = np.min(padded_embeddings, axis=0)
    distances = cosine_similarity(avg_embedding, candidate_embeddings.cpu())
    torch.cuda.empty_cache()
    return [candidates[index] for index in distances.argsort()[0][::-1][-top_k:]]



# Main pipeline
def main(args):
    data = get_data(args.n)
    articles = process_data(data)
    rank_articles=[]
    for i in tqdm(articles, desc="featurizing articles"):
        foreparts=str(i).split(',')[:2]  # location and date
        meat="".join(str(i).split(',')[2:-3])  # text
        aftparts=str(i).split(',')[-3:]  # org, per, theme
        try:
            cc=featurize_stories(str(i), top_k = args.f, max_len=512)
            rank_articles.append([foreparts,cc])
        except Exception as e:
            logging.error(f"Failed to process article: {e}")
    with open('DOTS/output/'+args.o, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rank_articles)

if __name__ == "__main__":
    main(args)
