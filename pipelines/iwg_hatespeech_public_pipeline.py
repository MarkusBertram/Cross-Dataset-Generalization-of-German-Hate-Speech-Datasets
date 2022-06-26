import json
import pickle
import csv
from pathlib import Path

data_path = Path(__file__).resolve().parents[1] / 'data' / 'iwg_hatespeech_public'

train = data_path / "german hatespeech refugees.csv"

#unlabeled = data_path + "germeval2018.test.txt"
    
def get_labeled_data():  
    dset_list = list()     

    with open(train, 'r', encoding="utf-8") as file:
        reader=csv.reader(file)

        for row in reader:
            entry=dict()
            entry['text'] = row[0]
            entry['label'] = row[1]
            #entry['fine-grained_label'] = row[3]
            dset_list.append(entry)
    #get unlabeled
    #read_data = read_file(unlabeled)
    #full_data["unlabeled"] = read_data

    return dset_list[1:]

def get_data_binary():
    binary_data = get_labeled_data()
    for entry in binary_data:
        if entry["label"] == "YES":
            entry["label"] = "abusive"
        else:
            entry["label"] = "neutral"
    return binary_data