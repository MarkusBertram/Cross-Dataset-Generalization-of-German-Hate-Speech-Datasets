#import shapX as shap
import transformers
import torch
import datasets
import yaml
import numpy as np
import scipy as sp
import pandas as pd
import shap
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime
import datasets
import os
import scipy as sp
import gc
import time
import torch
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import re
import unicodedata
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, BatchEncoding, Trainer, TrainingArguments, AdamW, get_scheduler
from utils.utils import fetch_import_module
from pipelines import utils_pipeline
from time import gmtime, strftime
from tqdm import tqdm
from datasets import concatenate_datasets,Dataset,ClassLabel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sys
from torch.utils.data import Dataset, TensorDataset, DataLoader
import subprocess
from accelerate import Accelerator
os.environ['MKL_THREADING_LAYER'] = 'GNU'
from sklearn.model_selection import train_test_split
from pathlib import Path

def predict(x):
    val = []
    for record in x:
        inputs =  tokenizer(record, return_tensors="pt")
        labels = torch.tensor([1]).unsqueeze(0)
        outputs = model(**inputs, labels=labels)
        m = torch.nn.Softmax(dim=1).cuda()
        # softmax the logits
        softmaxed = m(outputs.logits).detach().cpu().numpy()
        # get the probaility for the positive class (hate)
        val.append(softmaxed[0][1])
    return np.array(val)

def calculateError(x, dataset_names):
    error = 0
    for dataset in dataset_names:
        if x['Labels'] != x[dataset]:
            error += 1
    return error

def shap_explanations(d_names):
    path_models = Path('./tmp2/models/')
    path_combined_test_set = Path('./tmp2/datasets/combined_test')
    model_name= 'deepset/gbert-base'
    global dataset_names
    global tokenizer
    global model
    dataset_names = d_names
    # tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    # load combined test data set
    combined_test_dataset = datasets.load_from_disk(path_combined_test_set)
    #selected_test_data = combined_test_dataset[:20]

    df = pd.DataFrame(combined_test_dataset['label'],columns =['Labels'])

    for i,dataset_name in enumerate(tqdm(dataset_names)):
        predictions = []
        # load model
        path_to_model = path_models / '{}_{}_model'.format(i,dataset_name)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(path_to_model)
        # predict test set
        for record in combined_test_dataset:
            inputs =  tokenizer(record['text'], return_tensors="pt",truncation=True, max_length=512)
            labels = torch.tensor([1]).unsqueeze(0)
            outputs = model(**inputs, labels=labels)

            m = torch.nn.Softmax(dim=1).cuda()
            # softmax the logits
            softmaxed = m(outputs.logits).detach().cpu().numpy()
            # get the probaility for the positive class (hate)
            if softmaxed[0][1] >= 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
                
        df[dataset_name] = predictions

    df['Error'] = df.apply(lambda x: calculateError(x, dataset_names), axis=1)

    number_of_error = 2
    selected_test_data = []
    for label in range(2):
        errorLabel = 0 if label == 1 else 1
        for dataset in dataset_names:
            for i in range(5):
                try:
                    index = df[(df['Labels'] == label) & (df['Error'] == number_of_error) & (df[dataset] == errorLabel)].index[i]
                    selected_test_data.append(combined_test_dataset[int(index)]['text'])
                except:
                    break

    shap_values_of_models = []
    for i,dataset_name in enumerate(dataset_names):
        path_to_model = path_models / '{}_{}_model'.format(i,dataset_name)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(path_to_model)
        
        explainer = shap.Explainer(predict, masker=tokenizer)

        shap_values = explainer(selected_test_data)
        
        shap_values_of_models.append(shap_values)

    return shap_values_of_models