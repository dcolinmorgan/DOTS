## from https://huggingface.co/docs/transformers/training
# import spacy
import os,sys,torch,evaluate, numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, load_dataset #, train_test_split
model_name = "bert-base-uncased"
# model = AutoModel.from_pretrained(model_name)
# nlp = spacy.load('en_core_web_sm')
n_gram_range = (1, 2)
stop_words = "english"
embeddings=[]

device = torch.device('mps')
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
dataset = load_dataset("Dcolinmorgan/extreme-weather-news")

train_test_split = dataset['train'].train_test_split(test_size=0.2)  # here 'train' is artifact of loading
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']
train_tokens = tokenizer(train_dataset['story'], padding=True, return_tensors="pt")
test_tokens = tokenizer(test_dataset['story'], padding=True, return_tensors="pt")

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
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
    train_dataset= train_tokens,
    eval_dataset= test_tokens,
    compute_metrics=compute_metrics,
)
# config = {"repo_id": 'Dcolinmorgan'}

trainer.train()


# save locally
# model.save_pretrained("disaster-mlx-model", config=config)
# tokenizer.save_pretrained("disaster-mlx-model")

# push to the hub
# model.push_to_hub("Dcolinmorgan/disaster-mlx-model")#, config=config)

# model = model.from_pretrained("disaster-mlx-model")

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained('disaster-mlx-model')
