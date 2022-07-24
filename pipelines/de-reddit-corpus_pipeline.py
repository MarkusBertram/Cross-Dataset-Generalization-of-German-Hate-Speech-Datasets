import xml.etree.ElementTree as ET
from pathlib import Path
import sys
import json
data_path = Path(__file__).resolve().parents[1] / 'data' / 'de-reddit-corpus'

train = data_path / "german_subredditcorpus.json"

def get_labeled_data():
    with open(train, 'r', encoding = "utf-8") as f:
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