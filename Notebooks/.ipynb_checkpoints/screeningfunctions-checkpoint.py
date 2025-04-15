# Import necessary libraries
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

def get_embedding(text):
    '''
    The function takes in text, a tokenizer, model and arguments for the pretrained tokenizer.
    The functions process the parameters and returns embeddings
    '''
    inputs = tokenizer(text,padding=True,truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy

def calculate_centroid(embeddings):
    return np.mean(embeddings,axis = 0)