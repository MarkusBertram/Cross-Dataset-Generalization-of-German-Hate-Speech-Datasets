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
from sklearn import preprocessing
from utils.utils import (fetch_import_module, get_tweet_timestamp,
                         preprocess_text, print_data_example,
                         separate_text_by_classes)

pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000) 
sns.set_style("dark", {'axes.grid' : True, 'axes.linewidth':1})

if __name__ == "__main__":
    config = yaml.safe_load(open("settings/config.yaml"))
    # print("\n --- Downloading Stopwords ---")
    # nltk.download('stopwords')
    dataset_names = list(config['datasets'].keys())
    datasets = []
    for dset in dataset_names:
        dset_module = fetch_import_module(dset)
        datasets.append(dset_module.get_labeled_data())

    print("\n --- Calculating Cross-Dataset topic model... ---")