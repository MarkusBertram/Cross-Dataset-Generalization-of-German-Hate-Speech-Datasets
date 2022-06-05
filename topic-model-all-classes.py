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
#from torch.utils.datasets import Dataset, DataLoader
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
#from torch.utils.datasets import Dataset, TensorDataset
import processing.vocabulary_stats as vs
from utils import dataset_sampling, embedding_utils
#from torch.utils.datasets import Dataset
from processing.preprocessing_multilingual import preprocess_text
from sklearn import preprocessing
from utils.utils import fetch_import_module
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize  
from nltk.tokenize import wordpunct_tokenize
from spacy.tokenizer import Tokenizer
from spacy.lang.de import German
nltk.download('words')
import fasttext
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000) 
sns.set_style("dark", {'axes.grid' : True, 'axes.linewidth':1})

def nth_repl(s, sub, repl, n):
    find = s.find(sub)
    # If find is not -1 we have found at least one match for the substring
    i = find != -1
    # loop util we find the nth or we find no match
    while find != -1 and i != n:
        # find + 1 means we start searching from after the last match
        find = s.find(sub, find + 1)
        i += 1
    # If i is equal to n we found nth match so replace
    if i == n:
        return s[:find] + repl + s[find+len(sub):]
    return s

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
    
    print("\n --- Calculating Cross-Dataset topic model... ---")
    nltk.download('names')
    # constants
    n = 2000
    on_distribution=True
    components = 20
    file_suffix = 'all'
    embedding_path = "embeddings/cc.de.300.bin"
    now = datetime.now()
    fname = now.strftime("%Y-%m-%d_%H-%M-%S")
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
            if len(datasets[i][j]['text'].split()) < 1:
                del datasets[i][j]

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
    for t,l in zip(data_tweets,data_labels):
        samples,slabels = dataset_sampling.sample_tweets(t, l,n=n,on_distribution=on_distribution)
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
    cluwords_path = cluwords_launcher.generate_cluwords(sample_path, embedding_path, embedding_bin=True, components=components)
    
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
    
    
    results_dir = "./results/"+"topic_model_all_classes_"+now.strftime("%Y-%m-%d_%H-%M-%S")+"/"
    if os.path.exists(results_dir) == False:
        os.makedirs(results_dir)
    # path for storing image
    path_fig = results_dir
    #path_fig = "./results/"+strftime("%y%m%d", gmtime())+ "-" + "-".join(label_text).replace(" ","_") + file_suffix
    #colors
    palette = "colorblind"
    colors = sns.color_palette(palette, len(label_text))
    sns.set_palette(palette, len(label_text))
    sns.set_style("dark")
    # set font

    fig = plt.figure(figsize=(24,18))
    ax = fig.add_subplot(111)
    
    emb_x = tsne_embedded[:,0]
    emb_y = tsne_embedded[:,1]
    
    markers = ['x','^','*','+','p','>','2','v','H','<','>']
    sizes = [60,90,200,60,60,60,60,60,60,60,60]
    legend_elemens = []
    
    previous_end = 0
    for i,subset in enumerate(separators):
        start = previous_end
        end = start + subset
        previous_end = end
        
        s_sizes = [sizes[i]] * subset
        int_labels = labels[start:end]
        scatter = ax.scatter(x=emb_x[start:end], y=emb_y[start:end], 
                             s=s_sizes, c=[colors[i]], edgecolors='w', 
                             marker=markers[i],label=label_text[i])
    
    # topic centers
    s_sizes = [100] * components
    int_labels = labels[-1*components:]
    scatter2 = ax.scatter(x=emb_x[-1*components:], y=emb_y[-1*components:], 
                         s=s_sizes, c="black", edgecolors='w', marker='o', cmap=cm.jet,label="Topic center")
    
    texts = []
    label_counter = 0
    for i, text in zip(range(len(labels)-components,len(labels)), cluword_labels):
        label_counter += 1
        test_to_add = nth_repl(text," ","\n",15)
        test_to_add = nth_repl(text," ","\n",12)
        test_to_add = nth_repl(text," ","\n",9)
        test_to_add = nth_repl(test_to_add," ","\n",6)
        test_to_add = nth_repl(test_to_add," ","\n",3)
        test_to_add = f"(T{label_counter}) {test_to_add}"

        texts.append(plt.text(emb_x[i], emb_y[i], 
                              test_to_add,fontdict={'size': 16}, 
                              bbox=dict(boxstyle='round', fc='white', ec='w', alpha=0.7))) 
    # adjust text to avoid overlapping boxes
    iterations = adjust_text(texts,lim=5000)
    
    ax.legend( loc="upper right", title="Datasets")
    fig.savefig(path_fig + "content_tsne.pdf", bbox_inches='tight', dpi=300)
    fig.savefig(path_fig + "content_tsne.png", bbox_inches='tight', dpi=300)
    fig.savefig(path_fig + "content_tsne.eps", bbox_inches='tight', dpi=600)

    plt.show()