import json
import pickle
import csv
from pathlib import Path

data_path = Path(__file__).resolve().parents[1] / 'data' / 'germeval2018'

train = data_path / "germeval2018.training.txt"

test = data_path / "germeval2018.test.txt"

#unlabeled = data_path + "germeval2018.test.txt"

def read_file(filename):
    read_data=list()

    with open(filename, 'r', encoding="utf8") as file:
        reader=csv.reader(file, delimiter='\t', quotechar="'")

        for row in reader:
            entry=dict()
            entry['text'] = row[0]
            entry['label'] = row[1]
            entry['fine-grained_label'] = row[2]
            read_data.append(entry)

    return read_data
    
    
def get_labeled_data():  
    dset_list = list()     

    for dset in [train, test]:
        read_data = read_file(dset)
        dset_list.extend(read_data)

    return dset_list


def get_data_binary():
    binary_data = get_labeled_data()
    for entry in binary_data:
        del entry["fine-grained_label"]
        if entry["label"] == "OFFENSE":
            entry["label"] = "abusive"
        else:
            entry["label"] = "neutral"
    return binary_data