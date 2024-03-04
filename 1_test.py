## from https://huggingface.co/docs/transformers/training
# !pip install torch torchvision torchaudio
# !pip install datasets transformers --upgrade
# !pip3 install evaluate
# Import libraries
import os,sys,torch,evaluate, numpy as np
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, TrainingArguments, Trainer
sys.path.append(os.getcwd()) # Add current directory to PYTHONPATH if running as script
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans
from dirty_cat import TableVectorizer

# datasetA = load_dataset("yelp_review_full")
device = torch.device('mps')
dataset = load_dataset("melisekm/natural-disasters-from-social-media")

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
model.num_labels = 2

# import pandas as pd, matplotlib.pyplot as plt
# from cu_cat._dep_manager import deps
# deps.cuml=None
# AA = TableVectorizer().fit_transform(dataset['train']['event_type'])
# pca = PCA(2)
# pca_data = pd.DataFrame(pca.fit_transform(AA),columns=['PC1','PC2'])
# kmeans =KMeans(n_clusters=2).fit(AA)
# pca_data['cluster'] = pd.Categorical(kmeans.labels_)
# fig,ax = plt.subplots()
# scatter = ax.scatter(pca_data['PC1'], pca_data['PC2'],c=pca_data['cluster'],cmap='Set3',alpha=0.7)
# legend1 = ax.legend(*scatter.legend_elements(),
#                     loc="upper left", title="")
# ax.add_artist(legend1)
# plt.savefig('test')

# dataset = dataset.map(lambda examples: {'label': TableVectorizer().fit_transform(examples['label'])}, batched=True)
# dataset = dataset.map(lambda examples: {'label': KMeans(n_clusters=2).fit(examples['label']).labels_}, batched=True)
# dataset = dataset.map(lambda examples: {'event_type': 0 if examples['event_type']=='unknown' else 1}, batched=True)
# dataset = dataset.map(lambda examples: {**{k: v for k, v in examples.items() if k != 'event_type'}, 'event_type': 0 if examples['event_type']=='unknown' else 1}, batched=True)

def change_event_type(example):
    if example['event_type'] == 'unknown':
        example['label'] = 0
    else:
        example['label'] = 1
    return example

# Apply the function
dataset = dataset.map(change_event_type)
dataset.save_to_disk("disaster-tw.hf")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100000)).map(lambda examples:{'text': examples['text'], 'label': examples['label']},remove_columns=['target', 'SOURCE_FILE', 'tweet_id', 'filename', 'event_type', 'event_type_detail'])
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(20000)).map(lambda examples:{'text': examples['text'], 'label': examples['label']},remove_columns=['target', 'SOURCE_FILE', 'tweet_id', 'filename', 'event_type', 'event_type_detail'])

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
training_args = TrainingArguments(output_dir="test_trainer")

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch",save_strategy="epoch",save_total_limit=1,load_best_model_at_end=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= small_train_dataset,
    eval_dataset= small_eval_dataset,
    compute_metrics=compute_metrics,
)
config = {"repo_id": 'Dcolinmorgan'}

trainer.train("test_trainer/checkpoint-37500")


# save locally
# model.save_pretrained("disaster-mlx-model", config=config)
# tokenizer.save_pretrained("disaster-mlx-model")

# push to the hub
# model.push_to_hub("Dcolinmorgan/disaster-mlx-model")#, config=config)

# model = model.from_pretrained("disaster-mlx-model")

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained('disaster-mlx-model')
