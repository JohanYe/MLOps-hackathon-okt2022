from functools import cache
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import pickle
from google.cloud import storage
from run_main import api_reddit, api_twitter
import re
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from logic import get_reddits, reddits_to_df, get_tweets, tweets_to_df



FEATURES = "sepal_length  sepal_width  petal_length  petal_width".split()
labels = {
        0: "Anxiety",
        1: "BPD",
        2: "autism",
        3: "bipolar",
        4: "depression",
        5: "mentalhealth",
        6: "schizophrenia"
}

app = FastAPI()


def save_model(clf, model_path):
    clf_bytes = pickle.dumps(clf)
    bucket, path = re.match(r"gs://([^/]+)/(.+)", model_path).groups()
    storage.Client().bucket(bucket).blob(path).upload_from_string(clf_bytes)


@cache
def load_model(model_path):
    bucket, path = re.match(r"gs://([^/]+)/(.+)", model_path).groups()
    clf_bytes = storage.Client().bucket(bucket).blob(path).download_as_bytes()
    clf = pickle.loads(clf_bytes)
    return clf


class TrainRequest(BaseModel):
    dataset: str  # gs://path/to/dataset.csv
    features: List[str]
    target: str
    model: str  # gs://path/to/model.pkl


@app.post("/train")
def train_model(req: TrainRequest):
    dataset = pd.read_csv(req.dataset)
    X = dataset[req.features]
    y = dataset[req.target]
    clf = SVC().fit(X, y)
    save_model(clf, req.model)
    return "success"


class PredictRequest(BaseModel):
    accounts: List[str]

def remove_hyperlink(string):
    return re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)', "", string)

def remove_emoji(string):
    return re.sub("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", "", string)


def twitter(output, accounts, limit=100):
    tweets = []
    for acc in accounts:
        tweets += get_tweets(acc, limit)
    df = tweets_to_df(tweets)
    df.to_csv(output, index=False)

    

@app.post("/predict")
def predict(req: PredictRequest):
    twitter("output.csv", accounts=req.accounts, limit=100)
    data = pd.read_csv("output.csv")
    data['cleaned_text'] = data['text'].apply(remove_hyperlink)
    data['cleaned_text'] = data['cleaned_text'].apply(remove_emoji)
    data['cleaned_text'] = data['cleaned_text'].str.strip()
    data['cleaned_text'].replace('', np.nan, inplace=True)
    data.dropna(subset=['cleaned_text'], inplace=True)

    tweets = data.cleaned_text.values.tolist()

    model = AutoModelForSequenceClassification.from_pretrained("rabiaqayyum/autotrain-mental-health-analysis-752423172")
    tokenizer = AutoTokenizer.from_pretrained("rabiaqayyum/autotrain-mental-health-analysis-752423172")
    with torch.no_grad():
        inputs = tokenizer(tweets, return_tensors="pt", padding=True)
        outputs = model(**inputs)
    
        softmax = torch.nn.functional.softmax(outputs.logits).squeeze()
    data['max_label'] = torch.argmax(softmax, dim=1).numpy()
    data['label'] = data['max_label'].map(labels)
    data['prob'] = softmax[:,list(torch.argmax(softmax, dim=1).numpy())][:,0]
    data = data[data['prob'] > 0.5].reset_index()
    data.to_csv("model_predictions.csv", sep=";")
    return data

