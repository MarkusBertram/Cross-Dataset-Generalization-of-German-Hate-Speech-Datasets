import xml.etree.ElementTree as ET
from pathlib import Path
import sys
import json
data_path = Path(__file__).resolve().parents[1] / 'data' / 'de-reddit-corpus'

train = data_path / "german_subredditcorpus.json"

def get_labeled_data():
    with open(train, 'r') as f:
        dset_list = json.load(f)
    return dset_list

def get_data_binary():
    dset_list = get_labeled_data()
    for i in range(len(dset_list)):
        if dset_list[i]["label"] == "True":
            dset_list[i]["label"] = "abusive"
        elif dset_list[i]["label"] == "False":
            dset_list[i]["label"] = "neutral"
    return dset_list

# def get_labeled_data():
#     dset_list = []
#     count_false = 0
#     count_true = 0
#     file = open(train, encoding = "utf-8")
#     for event, ele in ET.iterparse(file):
#         if ele.tag == "comment" and ele.get("off") == "True" and count_true < 5000:
#             entry = dict()
#             entry['text'] = ele.text
#             entry['label'] = 'True'
#             dset_list.append(entry)
#             count_true += 1
#             ele.clear()
#         elif ele.tag == "comment" and ele.get("off") == "False" and count_false < 5000:
#             entry = dict()
#             entry['text'] = ele.text
#             entry['label'] = 'False'
#             dset_list.append(entry)
#             count_false += 1
#             ele.clear()

#     return dset_list

# def get_data_binary():
#     dset_list = []
    
#     count_false = 0
#     count_true = 0
#     file = open(train, encoding = "utf-8")
#     for event, ele in ET.iterparse(file):
#         if ele.tag == "comment" and ele.get("off") == "True" and count_true < 5000:
#             entry = dict()
#             entry['text'] = ele.text
#             entry['label'] = 'abusive'
#             dset_list.append(entry)
#             count_true += 1
#             ele.clear()
#         elif ele.tag == "comment" and ele.get("off") == "False" and count_false < 5000:
#             entry = dict()
#             entry['text'] = ele.text
#             entry['label'] = 'neutral'
#             dset_list.append(entry)
#             count_false += 1
#             ele.clear()
#     return dset_list