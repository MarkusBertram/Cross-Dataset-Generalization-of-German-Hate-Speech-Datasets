import json
import pickle
import csv
from pathlib import Path

data_path = Path(__file__).resolve().parents[1] / 'data' / 'covid_2021'

train = data_path / "covid_2021_dataset.csv"

#unlabeled = data_path + "germeval2018.test.txt"
    
def get_labeled_data():  
    dset_list = list()     

    with open(train, 'r', encoding="utf8") as file:
        reader=csv.reader(file)

        for row in reader:
            entry=dict()
            entry['text'] = row[0]
            entry['label'] = row[1]
            dset_list.append(entry)

    return dset_list[1:]

def get_data_binary():
    binary_data = get_labeled_data()
    for entry in binary_data:
        if entry["label"] == "abusive":
            entry["label"] = "abusive"
        else:
            entry["label"] = "neutral"
    return binary_data