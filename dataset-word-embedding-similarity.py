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
import torch
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
from datasets import Dataset
from utils.utils import (fetch_import_module, get_tweet_timestamp,
                         preprocess_text, print_data_example,
                         separate_text_by_classes)

pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000) 
sns.set_style("dark", {'axes.grid' : True, 'axes.linewidth':1})




def encode_labels(y_labels):
    encoding = dict()
    max_label = 0
    y_encoded = list()
    for entry in y_labels:
        if entry not in encoding:
            encoding[entry] = max_label
            max_label += 1
        y_encoded.append(encoding[entry])
    return y_encoded, encoding

def plot_embedding_annotate(tsne_embedded, axis_labels,dataset_names,palette = "colorblind"):
    now = datetime.now()
    results_dir = "./results/"+"word_embeddings_"+now.strftime("%Y%m%d-%H%M%S")+"/"
    if os.path.exists(results_dir) == False:
        os.makedirs(results_dir)
    # path for storing image
    path_fig = results_dir
    colors = sns.color_palette(palette, len(dataset_names))
    sns.set_palette(palette, len(dataset_names))
    
    emb_x = tsne_embedded[:,0]
    emb_y = tsne_embedded[:,1]
    
    int_labels = labels

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    
    scatter = ax.scatter(x=emb_x, y=emb_y, s=50, edgecolors='w', marker='o',c=colors)
    
    texts = []
    for j in range(len(dataset_names)):
        texts.append(plt.text(emb_x[j], emb_y[j], dataset_names[j],fontdict={'color':colors[j],'size': 12}))
    # offset = 0
    # texts = []
    # for i, number_of_labels in enumerate(labels_count):
    #     start = offset
    #     end = offset + number_of_labels
    #     scatter = ax.scatter(x=emb_x[start:end], y=emb_y[start:end], s=50, edgecolors='w', marker='o',c=[colors[i]])
        # for j in range(start,end):
        #     texts.append(plt.text(emb_x[j], emb_y[j], annotation_text[j],fontdict={'color':colors[i],'size': 12}))
        # offset = end

    iterations = adjust_text(texts,lim=2000,arrowprops=dict(arrowstyle='-', color='grey'))
    print(iterations)
    handles, labels = scatter.legend_elements()
    ax.set_xlabel('standardized PC1 ({:.2%} explained var.)'.format(axis_labels[0]))
    ax.set_ylabel('standardized PC2 ({:.2%} explained var.)'.format(axis_labels[1]))
    
    fig.savefig(path_fig + "-vocab_inter-intra-class-sim.pdf", bbox_inches='tight', dpi=300)
    fig.savefig(path_fig + "-vocab_inter-intra-class-sim.png", bbox_inches='tight', dpi=300)
    fig.savefig(path_fig + "-vocab_inter-intra-class-sim.eps", bbox_inches='tight', dpi=600)
    return fig

# tokenize datasets
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=512)

def transform_to_embed_sentence(dataset, model, tokenizer):
    dataset_tag_embedding = list()
    input_ids = []
    attention_masks = []
    labels = []
    
    for sentence in dataset:
        if not isinstance(sentence['text'], str):
            continue
        sentence["text"] = preprocessing_multilingual.clean_text(sentence['text'])

    df = pd.DataFrame(dataset)
    df.drop(["label", "fine-grained_label"], axis=1, inplace=True)
    ds = Dataset.from_pandas(df) 
    del dataset
    del df
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("----Tokenizing----")
    tokenized_ds = ds.map(tokenize, batched=True, batch_size=len(ds))
    tokenized_ds = tokenized_ds.remove_columns("text")
    #tokenized_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    tokenized_ds = tokenized_ds.with_format("torch", device = device)
    
    #dataloader = DataLoader(tokenized_ds, batch_size=32, num_workers=4)
    data = DataLoader(tokenized_ds, batch_size = 16, shuffle = False)
    accelerator = Accelerator()
    device = accelerator.device
    model, data = accelerator.prepare(model, data)
    print("----Calculating Embeddings----")
    with torch.no_grad():

        for x in data:
            outputs = model(input_ids = x["input_ids"], attention_mask = x['attention_mask'], output_hidden_states=True)
            cls_embeddings = outputs.hidden_states[0][:, 0]
            dataset_tag_embedding.append(cls_embeddings.detach().cpu().numpy())

    averaged_tag_embeddings = np.mean(np.concatenate(dataset_tag_embedding,axis=0),axis = 0)

    return averaged_tag_embeddings#, tag_labels

if __name__ == "__main__":
    config = yaml.safe_load(open("settings/config.yaml"))
    # print("\n --- Downloading Stopwords ---")
    # nltk.download('stopwords')
    dataset_names = list(config['datasets'].keys())
    #set_start_method("spawn")
    datasets = []
    for dset in dataset_names:
        dset_module = fetch_import_module(dset)
        datasets.append(dset_module.get_labeled_data())

    print("\n --- Calculating Word embedding based inter- and intra-dataset class similarity... ---")
    #config = AutoConfig.from_pretrained("deepset/gbert-base")
    model =  AutoModelForMaskedLM.from_pretrained("deepset/gbert-base")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-base")
    
    dataset_embeddings = []
    dataset_labels = []
    averaged_tag_embeddings = []
    tag_labels = []
    labels_count = []
    data_sets = []
    for dataset in datasets:
        dset = transform_to_embed_sentence(dataset, model, tokenizer)
        data_sets.append(dset)

    # for i,dataset in enumerate(datasets):
    #     embedding, labels = transform_to_embed_sentence(dataset, dataset_names[i], model, tokenizer)
    #     labels_count.append(len(labels))
    #     averaged_tag_embeddings += embedding
    #     tag_labels +=labels
    averaged_tag_embeddings = np.asarray(data_sets)
    n_averaged_tag_embeddings = np.nan_to_num(averaged_tag_embeddings)
    # set labels
    # y_encode, label_encoding = encode_labels(tag_labels)
    # label_text = label_encoding.keys()
    
    # apply PCA
    pca = PCA(n_components=2)
    pca_tag_embeddings = pca.fit_transform(n_averaged_tag_embeddings)
    
    # plot PCA results
    fig = plot_embedding_annotate(pca_tag_embeddings, pca.explained_variance_ratio_,dataset_names)

