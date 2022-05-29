import os
import pickle
import re
import warnings
from collections import defaultdict
from datetime import datetime
from math import log
from time import gmtime, strftime
import sys
from processing import cluwords_evaluation
from tools.cluwords import cluwords_launcher
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
from torch.utils.datasets import Dataset, DataLoader
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
from torch.utils.datasets import Dataset, TensorDataset
import processing.vocabulary_stats as vs
from utils import dataset_sampling, embedding_utils
from torch.utils.datasets import Dataset
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

def tweet_cleaner(tweets,m_names,f_names,extended_vocab ):
    c_samples = list()
    for tweet in tweets:
        new_tweet = list()
        for word in tweet:
            word = re.sub(r'[^\w\s]','',word)
            if len(word) > 4 or word in extended_vocab and not word.isnumeric():
                if word not in m_names and word not in f_names:
                    new_tweet.append(word)
        c_samples.append(new_tweet)
    return c_samples

def get_cluword_labels(cluword_path, num_topics):
    with open(cluword_path, 'r') as f:
        endl = num_topics + 2
        cluword_text = f.readlines()[2:endl]
    cluword_labels = list()
    for entry in cluword_text:
        cluword_labels.append(entry.rstrip())
    return cluword_labels

if __name__ == "__main__":
    config = yaml.safe_load(open("settings/config.yaml"))

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
    
    print("\n --- Calculating Cross-Dataset topic model... ---")

    # constants
    n = 2000
    on_distribution=False
    components = 20
    file_suffix = 'all'
    fname = strftime("%y%m%d", gmtime()) + "-" + "-".join(dataset_names).replace(" ","_")
    result_file = 'tools/cluwords/cluwords/multi_embedding/results/{}/matrix_w.txt'.format(fname)
    cluword_path = 'tools/cluwords/cluwords/multi_embedding/results/{}/result_topic_10.txt'.format(fname)
    sample_path = ''+fname
    
    # precleaning of datasets
    nlp = German()
    tokenizer = nlp.tokenizer

    with open("german_stop_words.txt", 'r', encoding = 'utf8') as f:
        stop_words = f.read().splitlines()
    
    for i,dataset in enumerate(datasets):
        for j,tweet in enumerate(dataset):
            datasets[i][j]['text'] = preprocessing_multilingual.preprocess_text(tweet['text'], tokenizer, stop_words)
    
    # sampling datasets
    ## splitting datasets
    data_tweets = []
    data_labels = []
    for dataset in datasets:
        tweets,labels =dataset_sampling.prepare_dataset(dataset) 
        data_tweets.append(tweets)
        data_labels.append(labels)
    
    ## actual sampling
    data_samples = []
    data_slabels = []
    for t,l,e in zip(data_tweets,data_labels,exclude):
        samples,slabels = dataset_sampling.sample_tweets(t, l, exclude_labels=e,n=n,on_distribution=on_distribution)
        data_samples.append(samples)
        data_slabels.append(slabels)
    
    # separators and merging samples
    separators = []
    samples = []
    for sample in data_samples:
        separators.append(len(sample))
        samples += sample
        
    # topic modeling cluwords
    #cluwords_evaluation.save_samples(samples, fname)
    
    ## cleaning datasets for topic modeling
    m_names = set(w.lower() for w in nltk.corpus.names.words('male.txt'))
    f_names = set(w.lower() for w in nltk.corpus.names.words('female.txt'))
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    extended_vocab = english_vocab | {'lol', 'rofl', 'lmao', 'yeah', 'wtf', 'omg', 'btw', 'nope'}
    
    samples = tweet_cleaner(samples,m_names,f_names,extended_vocab )
    cluwords_evaluation.save_samples(samples, sample_path)
    
    ## generating topic model
    ### default parameters: embedding_bin=False, threads=4, components=20, algorithm='knn_cosine'
    cluwords_path = cluwords_launcher.generate_cluwords(sample_path, embedding_path, components=components)
    
    ## loading topics from file
    
    n_components = components
    with open(result_file, 'r') as f:
        full_data = pd.read_csv(f, delimiter=' ', header=None)
    full_data = full_data.drop(n_components, axis=1)
    
    topics_per_dataset = cluwords_evaluation.separate_by_datasets(full_data, separators)
    
    data_topics_max = []
    data_topics_t = []
    data_tsne = []
    for i in range(len(datasets)):
        data_topics_max.append(cluwords_evaluation.topic_by_local_max(topics_per_dataset[i], n_components))
        data_topics_t.append(cluwords_evaluation.topic_by_threshhold(topics_per_dataset[i], n_components, threshold=0))
        data_tsne.append(topics_per_dataset[i])
            
    # TSNE projection for topics
    all_data = pd.concat(data_tsne)
    
    pure_topics = list()
    for i in range(components):
        a = np.zeros(components)
        a[i] = 1
        pure_topics.append(np.asarray(a))
    pure_topics = np.asarray(pure_topics)
    
    normalized_data = list()
    for i in range(len(all_data)):
        a = all_data.iloc[i] / sum(all_data.iloc[i])
        normalized_data.append(np.asarray(a))
    normalized_data = np.asarray(normalized_data)
    
    normalized_data = np.nan_to_num(normalized_data)
    topic_data = np.concatenate([normalized_data, pure_topics])
    pure_topic_data = np.concatenate([normalized_data, pure_topics])
    tsne_embedded = TSNE(n_components=2).fit_transform(pure_topic_data)
    
    labels = []
    label_text = []
    for i in range(len(datasets)):
        labels += [i] * separators[i]
        label_text.append(dataset_names[i])
    labels += [len(datasets)] * components
    label_text.append('Topic Centers')
    
    #Read in the cluwords for the topics to display
    cluword_labels = get_cluword_labels(cluword_path, components)
    plot_tsne_embedding_annotate(tsne_embedded, 
                                 labels, 
                                 label_text, 
                                 cluword_labels,
                                 separators,
                                 file_suffix,
                                 components=components,
                                 language=language).show()