from functools import cache
from typing import List

import numpy as np
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


def predict(twitter_account):
    twitter("output.csv", accounts=twitter_account, limit=100)
    data = pd.read_csv("output.csv")
    data['cleaned_text'] = data['text'].apply(remove_hyperlink)
    data['cleaned_text'] = data['cleaned_text'].apply(remove_emoji)
    data['cleaned_text'] = data['cleaned_text'].str.strip()
    data['cleaned_text'].replace('', np.nan, inplace=True)
    data.dropna(subset=['cleaned_text'], inplace=True)

    tweets = data.cleaned_text.values.tolist()

    model = AutoModelForSequenceClassification.from_pretrained("rabiaqayyum/autotrain-mental-health-analysis-752423172")
    tokenizer = AutoTokenizer.from_pretrained("rabiaqayyum/autotrain-mental-health-analysis-752423172")
    inputs = tokenizer(tweets, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    with torch.no_grad():
        softmax = torch.nn.functional.softmax(outputs.logits).squeeze()
    data['max_label'] = torch.argmax(softmax, dim=1).numpy()
    data['label'] = data['max_label'].map(labels)
    data['prob'] = softmax[:,list(torch.argmax(softmax, dim=1).numpy())][:,0]
    #data = data[data['prob'] > 0.5].reset_index()
    data.to_csv("model_predictions.csv", sep=";")
    #return data


if __name__ == "__main__":
    print(predict(twitter_account=['elonmusk', 'ylecun', 'POTUS']))