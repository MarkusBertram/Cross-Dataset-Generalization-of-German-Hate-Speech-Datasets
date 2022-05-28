import os
import pickle
import re
import warnings
from collections import defaultdict
from datetime import datetime
from math import log
from time import gmtime, strftime
import sys
from pprint import pprint
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from accelerate import Accelerator
from adjustText import adjust_text
from gensim import corpora, models, similarities
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from matplotlib import cm
from torch.utils.data import Dataset, DataLoader
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import SCORERS
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
import torch
import processing.basic_stats
import processing.emoji as emoji
import processing.preprocessing_multilingual as preprocessing_multilingual
import processing.user_stats
from torch.utils.data import Dataset, TensorDataset
import processing.vocabulary_stats as vs
from utils import dataset_sampling, embedding_utils
from torch.utils.data import Dataset
from processing.preprocessing_multilingual import preprocess_text
from sklearn import preprocessing
from utils.utils import fetch_import_module
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize  
from nltk.tokenize import wordpunct_tokenize
from spacy.tokenizer import Tokenizer
from spacy.lang.de import German
nltk.download('punkt')
nltk.download('stopwords')

pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000) 
sns.set_style("dark", {'axes.grid' : True, 'axes.linewidth':1})

def getMatrix(X):
    cv = CountVectorizer(lowercase = False,
                         max_df=0.95, 
                         min_df=2,                 
                         max_features=10000, 
                         binary=True)
    
    X_vec = cv.fit_transform(X)
    words = cv.get_feature_names()
    return X_vec,words

def getPmisPerClass(X_vec,Y,words):
    # get labels
    labels = set(Y)
    # create empty dict for results
    pmis_per_class = dict()
    for label in labels:
        pmis_per_class[label] = dict()

    X_matrix = np.array(X_vec.toarray())
    Y = np.array(Y)
    for i in range(len(X_matrix[0,:])):
        pmis = []
        for label in labels:
            p_label = np.sum(Y == label) / len(Y)
            select_y = Y == label
            column = X_matrix[:,i]
            select_column = column[select_y]
            p_label_x = np.sum(select_column) / len(select_column)
            if p_label_x <= 0:
                pass
                #pmis_per_class[label][words[i]] = 0
            else:
                pmi = log((p_label_x/p_label))
                pmis_per_class[label][words[i]] = pmi
    return pmis_per_class

if __name__ == "__main__":
    config = yaml.safe_load(open("settings/config.yaml"))
    # print("\n --- Downloading Stopwords ---")
    # nltk.download('stopwords')
    dataset_names = list(config['datasets'].keys())
    datasets, exclude = [], []
    
    for dset in dataset_names:
        dset_module = fetch_import_module(dset)
        datasets.append(dset_module.get_labeled_data())
        non_hate_labels = config['datasets'][dset]['non-hate-label']
        if isinstance(non_hate_labels, str):
            exclude.append(non_hate_labels)
        elif isinstance(non_hate_labels, list):
            exclude.append(non_hate_labels)
        else:
            pass
    
    
    print("\n --- Calculating PMI-based word ranking for classes... ---")
    nlp = German()
    tokenizer = nlp.tokenizer

    with open("german_stop_words.txt", 'r', encoding = 'utf8') as f:
        stop_words = f.read().splitlines()
    #stop_words = nlp.Defaults.stop_words
    
    n = 10
    
    content_table = dict()
    for dataset,dataset_name,exclude_labels in zip(datasets,dataset_names,exclude):
        X = [preprocess_text(x['text'], tokenizer, stop_words) for x in dataset]
        Y = [x['label'] for x in dataset]

        X_vec,words = getMatrix(X)

        pmis_per_class = getPmisPerClass(X_vec,Y,words)
        for label in pmis_per_class.keys():
            if label in exclude_labels:
                continue
            sorted_dict = dict(sorted(pmis_per_class[label].items(), key=lambda item: item[1],reverse=True))
            column = []
            for i,word in enumerate(sorted_dict.keys()):
                if i >= n:
                    break
                column.append(word)

            content_table[dataset_name + " - " + label] = column

    pmi_dict = pd.DataFrame.from_dict(content_table)

    print(pmi_dict)

    df2 = pmi_dict.T
    df2['Words with highest PMI'] = df2[0] + ", " + df2[1] + ", " + df2[2] + ", " + df2[3] + ", " + df2[4] + ", " + df2[5] + ", " + df2[6] + ", " + df2[7] + ", " + df2[8] + ", " + df2[9]
    df3 = df2.copy()
    df3 = df3.drop([0,1,2,3,4,5,6,7,8,9], axis = 1) 
    print(df3.to_latex())