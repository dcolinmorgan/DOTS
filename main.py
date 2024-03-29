import argparse
from tqdm import tqdm
from datetime import datetime
import numpy as np, pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModel, AutoTokenizer
import torch, spacy,nltk,subprocess, json, requests,string,csv,logging,os
import graphistry, umap

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


from DOTS.feat import chunk_text, featurize_stories
from DOTS.scrape import get_OS_data, get_google_news, get_massive_OS_data, get_npr_news, scrape_lobstr
from DOTS.pull import process_hit, process_data, pull_data, process_response, pull_lobstr_gdoc

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


def _input():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Process OS data for dynamic features.')
    parser.add_argument('-n', type=int, default=100, help='Number of data items to get')
    parser.add_argument('-f', type=int, default=5, help='Number of features per item to get')
    parser.add_argument('-o', type=str, default='dots_feats.csv', help='Output file name')
    # parser.add_argument('-p', type=int, default=1, help='Parallelize requests')
    # parser.add_argument('-t', type=int, default=1, help='Scroll Timeout in minutes, if using "d=1" large data set')
    parser.add_argument('-d', type=int, default=4, help='0 for a small amount, 1 for large, 2 for google news, 3 for lobstr, 4 for test_gnews')
    # parser.add_argument('-e', type=datetime, default=20231231, help='end date')
    args, unknown = parser.parse_known_args()
    return args

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

args = _input()

# Main pipeline
def main(args):
    if args.d == 0:
        # data = get_OS_data(args.n)
        # articles = process_data(data)
        response = get_OS_data(args.n)
        hits = response["hits"]["hits"]
        articles = pull_data(hits)
        # articles = process_response(data)
        dname = 'small0_'
    elif args.d == 1:
        response, client = get_massive_OS_data(args.t)
        pagination_id = response["_scroll_id"]
        hits = response["hits"]["hits"]
        articles = []
        while len(hits) != 0 and len(articles2) < args.n:
            response = client.scroll(
                scroll=str(args.t)+'m',
                scroll_id=pagination_id
                    )
            hits = response["hits"]["hits"]
            # article = process_data(response)
            articles.append(hits)
            articles2 = [item for sublist in articles for item in sublist]
        articles = [item for sublist in articles for item in sublist]
        dname = 'large1_'
    elif args.d == 2:
        articles = get_google_news('disaster')
        dname = 'google2_'
    elif args.d == 3:
        articles = pull_lobstr_gdoc()
        dname = 'lobstr3_'
    elif args.d == 4:
        response = get_test_gnews(100)
        hits = response["hits"]["hits"]
        articles = pull_data(hits)
        dname = 'test_gnews _'
    rank_articles = []
    if device == 'cuda':
        dataloader = DataLoader(data['text'], batch_size=1, shuffle=True, num_workers=4)
        RR = dataloader
    else:
        RR = articles
    for j,i in tqdm(enumerate(RR), total=len(RR), desc="featurizing articles"):
    # for i in tqdm(articles, desc="featurizing articles"):
        try:
            foreparts = str(i).split(',')[:2]  # location and date
        except:
            foreparts=None
        # meat="".join(str(j).split(',')[2:-3])  # text
        try:
            cc=featurize_stories(str(i), top_k = args.f, max_len=512)
            rank_articles.append([foreparts,cc])
            with open('DOTS/output/'+dname+args.o, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows([cc])
        except Exception as e:
            logging.error(f"Failed to process article: {e}")

    with open('DOTS/output/full_'+dname+args.o, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rank_articles)
        
    # flattened_list = [item for sublist in rank_articles for item in sublist]
    # import pandas as pd
    # data=pd.DataFrame(flattened_list)  # each ranked feature is a row
    # data.drop_duplicates(inplace=True)

    # object_columns = data.select_dtypes(include=['object']).columns
    # data[object_columns] = data[object_columns].astype(str)
    # g = graphistry.nodes(data)
    # g2 = g.umap()
    # g3 = g2.dbscan()
    # g3.encode_point_color('_dbscan',palette=["hotpink", "dodgerblue"],as_continuous=True).plot()

    

if __name__ == "__main__":
    main(args)
