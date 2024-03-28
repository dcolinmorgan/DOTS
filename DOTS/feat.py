# conda create -n DT ipykernel 
# python -m ipykernel install --user --name DT
# pip install torch bs4 transformers spacy numpy pandas scikit-learn scipy nltk
# import argparse
# from tqdm import tqdm
# from datetime import datetime
# import numpy as np, pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import CountVectorizer
# from transformers import AutoModel, AutoTokenizer
# import torch, spacy,nltk,subprocess, json, requests,string,csv,logging,os

# from .scrape import get_OS_data, get_massive_OS_data, get_google_news, scrape_lobstr  # need .scrape and .pull for production
# from .pull import process_hit, process_data, pull_data, pull_lobstr_gdoc

# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')


# Setup logging
# logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# # Setup argument parser
# parser = argparse.ArgumentParser(description='Process OS data for dynamic features.')
# # parser.add_argument('-n', type=int, default=100, help='Number of data items to get')
# parser.add_argument('-f', type=int, default=3, help='Number of features per item to get')
# parser.add_argument('-o', type=str, default='dots_feats.csv', help='Output file name')
# # parser.add_argument('-p', type=int, default=1, help='Parallelize requests')
# # parser.add_argument('-t', type=int, default=1, help='Scroll Timeout in minutes, if using "d=1" large data set')
# parser.add_argument('-d', type=int, default=1, help='0 for a small amount, 1 for large, 2 for google news, 3 for lobstr')
# # parser.add_argument('-e', type=datetime, default=20231231, help='end date')
# args, unknown = parser.parse_known_args()


# # Load models and tokenizers
# model_name = "distilroberta-base"
# model = AutoModel.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# # !python -m spacy download en_core_web_sm
# nlp = spacy.load('en_core_web_sm')

# # Define constants
# n_gram_range = (1, 2)
# stop_words = "english"
# embeddings=[]
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# Define functions
def chunk_text(text, max_len):
    tokens = nltk.word_tokenize(text)
    num_chunks = len(tokens) // max_len
    final_chunk_size = len(tokens) % max_len
    # logging.info(f"chunking artcile into {num_chunks} chunks")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
    candidate_tokens = {k: v.to(device) for k, v in (candidate_tokens).items()}
    candidate_embeddings = model(**candidate_tokens)["pooler_output"]
    if device == 'cuda':
        candidate_embeddings = candidate_embeddings.detach().to_numpy
    else:
        candidate_embeddings = candidate_embeddings.detach()
    chunks = chunk_text(text, max_len)  # use this to chunk better and use less padding thus less memory but also less affect from averging
    for chunk in chunks:
        text_tokens = tokenizer(chunk, padding=True, return_tensors="pt")
        # if device == 'cuda':
        text_tokens = {k: v.to(device) for k, v in (text_tokens).items()}
        text_embedding = model(**text_tokens)["pooler_output"]
        if device == 'cuda':
            text_embedding = text_embedding.detach().to_numpy()
        else:
            text_embedding = text_embedding.detach()
        embeddings.append(text_embedding)
    max_emb_shape = max(embedding.shape[0] for embedding in embeddings)
    padded_embeddings = [np.pad(embedding.cpu(), ((0, max_emb_shape - embedding.shape[0]), (0, 0))) for embedding in embeddings]
    avg_embedding = np.min(padded_embeddings, axis=0)
    distances = cosine_similarity(avg_embedding, candidate_embeddings.cpu())
    torch.cuda.empty_cache()
    return [candidates[index] for index in distances.argsort()[0][::-1][-top_k:]]


# # Main pipeline
# def main(args):
#     if args.d == 0:
#         data = get_OS_data(args.n)
#         articles = process_data(data)
#         # articles = process_response(data)
#         dname = 'small0_'
#     elif args.d == 1:
#         response, client = get_massive_OS_data(args.t)
#         pagination_id = response["_scroll_id"]
#         hits = response["hits"]["hits"]
#         articles = []
#         while len(hits) != 0 and len(articles2) < args.n:
#             response = client.scroll(
#                 scroll=str(args.t)+'m',
#                 scroll_id=pagination_id
#                     )
#             hits = response["hits"]["hits"]
#             # article = process_data(response)
#             articles.append(hits)
#             articles2 = [item for sublist in articles for item in sublist]
#         articles = [item for sublist in articles for item in sublist]
#         dname = 'large1_'
#     elif args.d == 2:
#         articles = get_google_news('disaster')
#         dname = 'google2_'
#     elif args.d == 3:
#         articles = pull_lobstr_gdoc()
#         dname = 'lobstr3_'
#     rank_articles = []
#     if device == 'cuda':
#         dataloader = DataLoader(data['text'], batch_size=1, shuffle=True, num_workers=4)
#         RR = dataloader
#     else:
#         RR = articles
#     for j,i in tqdm(enumerate(RR), total=len(RR), desc="featurizing articles"):
#     # for i in tqdm(articles, desc="featurizing articles"):
#         try:
#             foreparts = str(i).split(',')[:2]  # location and date
#         except:
#             foreparts=None
#         # meat="".join(str(j).split(',')[2:-3])  # text
#         try:
#             cc=featurize_stories(str(i), top_k = args.f, max_len=512)
#             rank_articles.append([foreparts,cc])
#             with open('DOTS/output/'+dname+args.o, 'a', newline='') as file:
#                 writer = csv.writer(file)
#                 writer.writerows([cc])
#         except Exception as e:
#             logging.error(f"Failed to process article: {e}")

#     with open('DOTS/output/full_'+dname+args.o, 'a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerows(rank_articles)

# if __name__ == "__main__":
#     main(args)
