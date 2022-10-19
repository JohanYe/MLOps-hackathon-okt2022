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


labels = {
        0: "Anxiety",
        1: "BPD",
        2: "autism",
        3: "bipolar",
        4: "depression",
        5: "mentalhealth",
        6: "schizophrenia"
}


def predict(tweet):
    model = AutoModelForSequenceClassification.from_pretrained("rabiaqayyum/autotrain-mental-health-analysis-752423172")
    tokenizer = AutoTokenizer.from_pretrained("rabiaqayyum/autotrain-mental-health-analysis-752423172")
    inputs = tokenizer(tweet, return_tensors="pt")
    outputs = model(**inputs)
    with torch.no_grad():
        softmax = torch.nn.functional.softmax(outputs.logits).squeeze()
        
    return {"prediction": labels[torch.argmax(softmax).item()],
            "prob": softmax[torch.argmax(softmax).item()]
            }



if __name__ == "__main__":
    tweet = 'I am bipolar'
    print(predict(tweet))
    tweet = 'I am depressed'
    print(predict(tweet))