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

def plot_embedding_annotate(tsne_embedded, labels, label_text, annotation_text,labels_count,axis_labels,dataset_names,palette = "colorblind"):
    path_fig = "./results/"+strftime("%y%m%d", gmtime())+ "-" + "-".join(dataset_names).replace(" ","_")
    colors = sns.color_palette(palette, len(labels_count))
    sns.set_palette(palette, len(labels_count))
    
    emb_x = tsne_embedded[:,0]
    emb_y = tsne_embedded[:,1]
    
    int_labels = labels

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    offset = 0
    texts = []
    for i, number_of_labels in enumerate(labels_count):
        start = offset
        end = offset + number_of_labels
        scatter = ax.scatter(x=emb_x[start:end], y=emb_y[start:end], s=50, edgecolors='w', marker='o',c=[colors[i]])
        for j in range(start,end):
            texts.append(plt.text(emb_x[j], emb_y[j], annotation_text[j],fontdict={'color':colors[i],'size': 12}))
        offset = end
        

    iterations = adjust_text(texts,lim=2000,arrowprops=dict(arrowstyle='-', color='grey'))
    print(iterations)
    handles, labels = scatter.legend_elements()
    ax.set_xlabel('standardized PC1 ({:.2%} explained var.)'.format(axis_labels[0]))
    ax.set_ylabel('standardized PC2 ({:.2%} explained var.)'.format(axis_labels[1]))
    
    fig.savefig(path_fig + "-vocab_inter-intra-class-sim.pdf", bbox_inches='tight', dpi=300)
    fig.savefig(path_fig + "-vocab_inter-intra-class-sim.png", bbox_inches='tight', dpi=300)
    fig.savefig(path_fig + "-vocab_inter-intra-class-sim.eps", bbox_inches='tight', dpi=600)
    return fig

def transform_to_embed_sentence(dataset, dset_name, model, tokenizer):
    dataset_tag_embedding = list()
    input_ids = []
    attention_masks = []
    labels = []
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # get sentence vector
    for sentence in dataset:
        if not isinstance(sentence['text'], str):
            continue
        text = preprocessing_multilingual.clean_text(sentence['text'])
        # tokenize text
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens = True,
            padding = 'max_length',
            max_length = tokenizer.model_max_length,
            truncation = True,
            return_attention_mask = True,
            return_tensors = 'pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        
        label = dset_name + "_" + str(sentence['label'])
        labels.append(label)

    le = preprocessing.LabelEncoder()
    targets = le.fit_transform(labels)
    targets = torch.as_tensor(targets)
    
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    stacked_input = torch.stack((input_ids, attention_masks), dim=1)
    
    tensordataset = TensorDataset(stacked_input, targets)
    data = DataLoader(tensordataset, batch_size = 16, shuffle = False)
    accelerator = Accelerator()
    device = accelerator.device
    model, data = accelerator.prepare(model, data)
    with torch.no_grad():
        for x, y in data:
            input_id = x[:, 0].to(device)
            attention_masks = x[:, 1].to(device)
            outputs = model(input_ids = input_id, attention_mask = attention_masks, output_hidden_states=True)

            cls_embeddings = outputs.hidden_states[0][:, 0]

            label = le.inverse_transform(y.detach().cpu().numpy()).tolist()

            # cls_embeddings = last_hidden_state
            #label = dset_name + "_" + str(le.inverse_transform(y.detach().cpu().numpy()))
            dataset_tag_embedding.append((cls_embeddings.detach().cpu().numpy(), label))
    
    tag_embeddings = dict()
    tag_count = defaultdict(int)
    
    for entry in dataset_tag_embedding:
        for i in range(len(entry[1])):
            if entry[1][i] in tag_embeddings:
                tag_embeddings[entry[1][i]] += entry[0][i]
            else:
                tag_embeddings[entry[1][i]] = entry[0][i]
            tag_count[entry[1][i]] += 1

    #average vectors
    averaged_tag_embeddings = list()
    tag_labels = list()
    for key, value in tag_embeddings.items():
        averaged_tag_embeddings.append(value / tag_count[key])
        tag_labels.append(key)
    
    return averaged_tag_embeddings, tag_labels

if __name__ == "__main__":
    config = yaml.safe_load(open("settings/config.yaml"))
    # print("\n --- Downloading Stopwords ---")
    # nltk.download('stopwords')
    dataset_names = list(config['datasets'].keys())
    datasets = []
    for dset in dataset_names:
        dset_module = fetch_import_module(dset)
        datasets.append(dset_module.get_labeled_data())

    print("\n --- Calculating Word embedding based inter- and intra-dataset class similarity... ---")
    config = AutoConfig.from_pretrained("deepset/gbert-base")
    model =  AutoModelForMaskedLM.from_pretrained("deepset/gbert-base", config = config)
    model.eval()
    #tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
    tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-base")
    
    dataset_embeddings = []
    dataset_labels = []
    averaged_tag_embeddings = []
    tag_labels = []
    labels_count = []
    for i,dataset in enumerate(datasets):
        embedding, labels = transform_to_embed_sentence(dataset, dataset_names[i], model, tokenizer)
        #dataset_embeddings.append(embedding)
        #dataset_labels.append(labels)
        labels_count.append(len(labels))
        averaged_tag_embeddings += embedding
        tag_labels +=labels
    
    print("\n embedding:\n")
    print(len(embedding))
    print(type(embedding))
    print(len(embedding[0]))
    print("\n labels:\n")
    print(len(labels))
    print(type(labels))
    print(len(labels[0]))

    # print("\n dataset_embeddings \n")
    # print(len(dataset_embeddings))
    # print(type(dataset_embeddings))
    # #print(dataset_embeddings[0])

    # print("\n dataset_labels \n")
    # print(len(dataset_labels))
    # print(type(dataset_labels))
    # #print(dataset_labels[0])

    print("\n labels_count \n")
    print(len(labels_count))
    print(type(labels_count))
    print(len(labels_count[0]))

    print("\n averaged_tag_embeddings \n")
    print(len(averaged_tag_embeddings))
    print(type(averaged_tag_embeddings))
    print(len(averaged_tag_embeddings[0]))

    print("\n tag_labels \n")
    print(len(tag_labels))
    print(type(tag_labels))
    print(len(tag_labels[0]))

    n_averaged_tag_embeddings = np.nan_to_num(averaged_tag_embeddings)
    #n_averaged_tag_embeddings = np.nan_to_num(dataset_embeddings) 
    # set labels
    y_encode, label_encoding = encode_labels(tag_labels)
    label_text = label_encoding.keys()
    
    # apply PCA
    pca = PCA(n_components=2)
    pca_tag_embeddings = pca.fit_transform(n_averaged_tag_embeddings)
    
    # plot PCA results
    fig = plot_embedding_annotate(pca_tag_embeddings, y_encode, list(tag_labels), list(tag_labels),labels_count,pca.explained_variance_ratio_,dataset_names)

