import json
import pickle
import csv
from pathlib import Path

data_path = Path(__file__).resolve().parents[1] / 'data' / 'telegram_gold'

train = data_path / 'annotated_dataset.txt'

def get_labeled_data():
    dset_list = []
    
    with open(train, 'r') as f:
        data_json = json.load(f)

    for message in data_json["messages"]:
        entry = dict()
        entry['text'] = message["text"]
        entry['label'] = message["gold_label"]
        dset_list.append(entry)
    return dset_list


def get_data_binary():
    binary_data = get_labeled_data()
    for entry in binary_data:
        if entry["label"] == "OFFENSIVE_ABUSIVE":
            entry["label"] = "abusive"
        else:
            entry["label"] = "neutral"
    return binary_data