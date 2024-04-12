
from tqdm import tqdm
from datetime import datetime
import numpy as np, pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModel, AutoTokenizer
import torch, spacy,nltk,subprocess, json, requests,string,csv,logging,os
import graphistry
from graphistry.features import search_model, topic_model, ngrams_model, ModelDict, default_featurize_parameters, default_umap_parameters


# # Load models and tokenizers
model_name = "distilroberta-base"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# !python -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm')

# # Define constants
n_gram_range = (1, 2)
stop_words = "english"
embeddings=[]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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


def g_feat(text, top_k=3, n_topics=42):
    ### need to parse the text to remove the extra characters quite heavily
    df=pd.Series(text).to_frame()
    object_columns = df.select_dtypes(include=['object']).columns
    df[object_columns] = df[object_columns].astype(str)
    df.columns=['text']
    df['text'] = df['text'].str.replace("', '", ' ')
    df['text'] = df['text'].str.replace('", "', ' ')
    df['text'] = df['text'].str.replace('" "', ' ')
    df['text'] = df['text'].str.replace("' '", ' ')
    df['text'] = df['text'].str.replace("\'", ' ')
    df['text'] = df['text'].str.replace("]", ' ')
    df['text'] = df['text'].str.replace("[", ' ')
    object_columns = df.select_dtypes(include=['object']).columns
    df[object_columns] = df[object_columns].astype(str)
    
    topic_model['n_topics'] = n_topics
    g=graphistry.nodes(df)
    g2 = g.umap(df,**topic_model)
    g2 = g2.dbscan(min_dist=2, min_samples=1)
    g3 = g2.transform_dbscan(df,return_graph=False)
    df2=pd.DataFrame(g2.get_matrix())

    max_index_per_row = df2.idxmax(axis=1)
    top_3_indices = max_index_per_row.value_counts().index[:top_k]
    top_3_indices
    
    return df2, top_3_indices


def gliner_feat(text, hits,labels=['disaster','weather',"avalanche",
        "biological","blizzard",
        "chemical","contamination","cyber",
        "drought",
        "earthquake","explosion",
        "fire","flood",
        "heat","hurricane",
        "infrastructure",
        "landslide",
        "nuclear",
        "pandemic","power",
        "radiological","riot",
        "sabotage",
        "terror","tornado","transport","tsunami",
        "volcano"]):
    from gliner import GLiNER
    model = GLiNER.from_pretrained("urchade/gliner_base")

    df = pd.DataFrame(columns=['Title','URL', 'Text', 'Label', 'Date','Location'])

    for hit,article in zip(hh[:100],text[:100]):
        entities = model.predict_entities(''.join(article), labels)
        for entity in entities:
            row = pd.DataFrame({'Title': [hit['_source']['metadata']['title']],
                                'URL': [hit['_source']['metadata']['link']], 
                                'Text': [entity["text"]], 
                                'Label': [entity["label"]]})
            df = pd.concat([df, row], ignore_index=True)
    df2 = df.groupby(['Title','URL','Label']).agg({'Text': ' '.join}).reset_index()
    df_pivot = df2.pivot(index=['Title','URL'], columns='Label', values='Text')
    df_pivot.reset_index(inplace=True)
    df_pivot = df_pivot.dropna(subset=df_pivot.columns[2:],how='all')
    return df_pivot
    
def count_gliner(df,label='disaster'):
    from collections import Counter
    lst = df.Text[df.Label=='earthquake']
    counter = Counter(lst)
    sorted_lst = sorted(lst, key=lambda x: -counter[x])
    return pd.unique(sorted_lst)

def gpy_gliner(df_pivot):
    g=graphistry.nodes(df_pivot.drop(['URL','Title'],axis=1))
    g2 = g.umap()  # df_pivot.drop(['URL','Title'],axis=1),**topic_model)
    g2 = g2.dbscan()  # min_dist=1, min_samples=3)
    # g3 = g2.transform_dbscan(df_pivot.drop(['URL','Title'],axis=1),return_graph=False)
    df2=pd.DataFrame(g2.get_matrix())

    max_index_per_row = df2.idxmax(axis=1)
    top_3_indices = max_index_per_row.value_counts().index[:10]
    top_3_indices
    return top_3_indices, g2, df2
