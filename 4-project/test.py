from functools import cache
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
#from run_main import api_reddit, api_twitter
import re
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from logic import get_reddits, reddits_to_df, get_tweets, tweets_to_df

labels = {
        0: "Anxiety",
        1: "BPD",
        2: "autism",
        3: "bipolar",
        4: "depression",
        5: "mentalhealth",
        6: "schizophrenia"
}

def twitter(output, accounts, limit=100):
    tweets = []
    for acc in accounts:
        tweets += get_tweets(acc, limit)
    df = tweets_to_df(tweets)
    df.to_csv(output, index=False)


def predict(tweet):
    data = pd.read_csv("output.csv")
    tweets = data.text.values.tolist()

    model = AutoModelForSequenceClassification.from_pretrained("rabiaqayyum/autotrain-mental-health-analysis-752423172")
    tokenizer = AutoTokenizer.from_pretrained("rabiaqayyum/autotrain-mental-health-analysis-752423172")
    inputs = tokenizer(tweets, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    with torch.no_grad():
        softmax = torch.nn.functional.softmax(outputs.logits).squeeze()
    
    print(torch.argmax(softmax, dim=1).shape)
    print(softmax.shape)
    print(softmax[torch.argmax(softmax, dim=1)].shape)
    data['max_label'] = torch.argmax(softmax, dim=1).numpy()
    #data['label'] = labels[data['max_label']]
    data['label'] = data['max_label'].map(labels)
    data['prob'] = softmax[torch.argmax(softmax, dim=1)]
    print(data.tail())

    #return {"prediction": labels[torch.argmax(softmax).item()],
    #        "prob": softmax[torch.argmax(softmax).item()]
    #        }



if __name__ == "__main__":
    twitter("output.csv", ['kanyewest'], 100)
    tweet = 'I am bipolar'
    print(predict(tweet))