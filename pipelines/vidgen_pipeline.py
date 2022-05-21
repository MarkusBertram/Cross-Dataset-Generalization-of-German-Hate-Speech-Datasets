import pandas as pd
from pathlib import Path
import csv
#data_path = PATH_EXPORT_PKL = "./data/Vidgen_2020/1_crawled/hs_AsianPrejudice_20kdataset_cleaned_with_user_ids.pkl"

data_path = Path(__file__).resolve().parents[1] / 'data' / 'Vidgen_2020' / 'hs_AsianPrejudice_20kdataset_cleaned_anonymized.tsv'

def shortenLabel(label):
    if label == "none_of_the_above":
        return "none"
    if label == "entity_directed_hostility":
        return "hostility"
    if label == "discussion_of_eastasian_prejudice":
        return "prejudice"
    if label == "entity_directed_criticism":
        return "cristism"
    if label == "counter_speech":
        return "counter"

def get_labeled_data():
    dset_list = list()
    filename = data_path

    with open(filename, 'r', encoding='latin-1') as file:
        reader=csv.reader(file, delimiter='\t')

        for row in reader:
            entry=dict()
            entry['text'] = row[4]
            entry['label'] = shortenLabel(row[3])
            #entry['fine-grained_label'] = row[2]
            dset_list.append(entry)

    #df = pd.read_pickle(data_path)
    # for index,row in df.iterrows():
    #     entry = dict()
    #     entry['text'] = row['text']
    #     entry['label'] = shortenLabel(row['expert'])
    #     dset_list.append(entry)
    return dset_list

def get_data_binary():
    dset_list = list()
    df = pd.read_pickle(data_path)
    for index,row in df.iterrows():
        entry = dict()
        entry['text'] = row['text']
        if row['expert']== "entity_directed_hostility":
            entry['label'] = 'abusive'
        else:
            entry['label'] = 'neutral'
        dset_list.append(entry)
    return dset_list

# def get_user_data():
#     dset_list = list()
#     df = pd.read_pickle(data_path)
#     for index,row in df.iterrows():
#         if row['user_id'] != '':
#             entry = dict()
#             entry['text'] = row['text']
#             entry['label'] = shortenLabel(row['expert'])
#             entry['id'] = row['tweet_id']
#             entry['user'] = dict()
#             entry['user']['id'] = row['user_id']
#             dset_list.append(entry)
#     return dset_list