from collections import defaultdict
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from gensim import corpora, models, similarities
from datetime import datetime
from utils.utils import fetch_import_module
from nltk.corpus import stopwords
import nltk
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000) 
sns.set_style("dark", {'axes.grid' : True, 'axes.linewidth':1})

# Cleaning up raw input text
def filter_corpus(corpus_input):
    filtered_corpus = [
                [token for token in document.lower().split() if token not in stop_words]
                for document in corpus_input
    ]
    corp_frequency = defaultdict(int)
    for text in filtered_corpus:
        for token in text:
            corp_frequency[token] += 1
    filtered_corpus = [
                        [token for token in text if corp_frequency[token] > 1]
                        for text in filtered_corpus
    ]
    return filtered_corpus

def getSimilarityScores(data):
    class_separated_dicts = separate_text_by_classes(data)
    labels = list(class_separated_dicts.keys())
    total_corpus = list()
    for label in labels:
        total_corpus.append(filter_corpus(class_separated_dicts[label]))
        
    ### CLASS SIMILARITIES
    dictionary = corpora.Dictionary([item for sublist in total_corpus for item in sublist])
    corpus = [dictionary.doc2bow(text) for text in [item for sublist in total_corpus for item in sublist]]
    #LSI similarity
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=16)
    index = similarities.MatrixSimilarity(lsi[corpus])

    similarity_scores = dict()
    for i, entry in enumerate(labels):
        scores = get_total_sim_score(i, total_corpus, labels, dictionary, lsi, index)
        similarity_scores[entry] = scores
    return similarity_scores

# summing over similarities
def get_total_sim_score(query, corpus, labels, dictionary, lsi, index):
    query_corpus = corpus[query]
    tally_scores = defaultdict(float)
    for entry in query_corpus:
        scores = get_sim_score(entry, corpus, labels, dictionary, lsi, index)
        for label, score in scores.items():
            tally_scores[label] += score
    for label, score in tally_scores.items():
        tally_scores[label] = score/len(query_corpus)
    return tally_scores

# get the corpus sorted by labels
def separate_text_by_classes(data):
    text_corpus = defaultdict(list)
    for entry in data:
        text = entry['text']
        label = entry['label']
        text_corpus[label].append(text)
    return text_corpus

# basic similarity scoring by cosine distance
def get_sim_score(query_i, corpus, labels, dictionary, lsi, index):
    query = dictionary.doc2bow(query_i)
    vec_lsi = lsi[query]
    sims = index[vec_lsi]
    all_scored = dict()
    index = 0
    for i, entries in enumerate(corpus):
        similarity_score = sum(sims[index:index+len(entries)])/len(entries)       
        tag = labels[i]
        all_scored[tag] = similarity_score
        index += len(entries)
    return all_scored

if __name__ == "__main__":
    config = yaml.safe_load(open("settings/config.yaml"))
    global stop_words
    with open("german_stop_words.txt", 'r', encoding = 'utf8') as f:
        stop_words = f.read().splitlines()
    dataset_names = list(config['datasets'].keys())
    datasets = []
    for dset in dataset_names:
        dset_module = fetch_import_module(dset)
        datasets.append(dset_module.get_labeled_data())

    print("\n --- Calculating LSI-based Intra-dataset class similarity... ---")
    title = ""
    rows = 2
    cols = 4
    width = 12
    height = 6
    sync_scaling=False
    cmap = "Blues"

    now = datetime.now()
    results_dir = "./results/"+"lsi_"+now.strftime("%Y%m%d-%H%M%S")+"/"
    if os.path.exists(results_dir) == False:
        os.makedirs(results_dir)
    # path for storing image
    path_fig = results_dir + "-".join(dataset_names).replace(" ","_")
    
    fig2 = plt.figure(constrained_layout=True,figsize=(width, height))
    fig2.suptitle(title,y=1.05,fontsize=16)
    spec2 = gridspec.GridSpec(ncols=cols, nrows=rows, figure=fig2)
    
    global_min = 1
    global_max = 0
    all_similarity_scores = []
    for dataset in datasets:
        scores = getSimilarityScores(dataset)
        all_similarity_scores.append(scores)
        for entry in scores.values():
            for ef in entry.values():
                global_min = min(global_min,ef)
                global_max = max(global_max,ef)
    m = 0
    ax = []

    for k in range(rows):
        for l in range(cols):
            if m < len(datasets):
                t_list = list()
                for entry in all_similarity_scores[m].values():
                    a_list = list()
                    for ef in entry.values():
                        a_list.append(ef)
                    t_list.append(a_list)

                ax.append(fig2.add_subplot(spec2[k, l]))
                #ax.append(fig2.add_subplot(spec2[k, l]))
                # define colors and style
                labels = all_similarity_scores[m].keys() 

                sheat = sns.heatmap(t_list, annot=True, fmt=".2f",
                                    ax=ax[m],square=False, label=dataset_names[m], 
                                    xticklabels=labels, yticklabels=labels,
                                    vmin=global_min, vmax=global_max,
                                    cmap=cmap,cbar=False)
                sheat.set_xticklabels(labels, rotation=45, ha='center')
                sheat.set_yticklabels(labels, rotation=0, ha='right')
                #ax[m].xaxis.set_label_text(dataset_names[m])
                ax[m].set_title(dataset_names[m])
                #ax.set_title(dataset_names[m])
                
                m += 1
    fig2.savefig(path_fig + "-vocab_intra-dataset-sim.pdf", bbox_inches='tight', dpi=300)
    fig2.savefig(path_fig + "-vocab_intra-dataset-sim.png", bbox_inches='tight', dpi=300)
    fig2.savefig(path_fig + "-vocab_intra-dataset-sim.eps", bbox_inches='tight', dpi=600)

    print("\n --- Calculation finished. Output is saved in results folder. ---")