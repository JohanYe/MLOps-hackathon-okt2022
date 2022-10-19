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
from transformers import AutoModelForSequenceClassification, AutoTokenizer


FEATURES = "sepal_length  sepal_width  petal_length  petal_width".split()


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
    accounts: List[dict] 

def find_hyperlink(string):
    return re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)', "", string)


@app.post("/predict")
def predict(req: PredictRequest):
    twitter("output.csv", accounts=twitter_account, limit=100)
    data = pd.read_csv("output.csv")
    data['cleaned_text'] = data['text'].apply(find_hyperlink)
    data['cleaned_text'].replace('', np.nan, inplace=True)
    data.dropna(subset=['cleaned_text'], inplace=True)

    tweets = data.text.values.tolist()
    tweets['text'].apply(find_hyperlink)

    model = AutoModelForSequenceClassification.from_pretrained("rabiaqayyum/autotrain-mental-health-analysis-752423172")
    tokenizer = AutoTokenizer.from_pretrained("rabiaqayyum/autotrain-mental-health-analysis-752423172")
    inputs = tokenizer(tweets, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    with torch.no_grad():
        softmax = torch.nn.functional.softmax(outputs.logits).squeeze()
    data['max_label'] = torch.argmax(softmax, dim=1).numpy()
    data['label'] = data['max_label'].map(labels)
    data['prob'] = softmax[:,list(torch.argmax(softmax, dim=1).numpy())][:,0]
    return data

