import csv
import pickle
from pathlib import Path

data_path = Path(__file__).resolve().parents[1] / 'data' / 'ihs_data'

train = data_path / 'iHS-data_anno.tsv'

unlabeled = data_path / 'tweets.csv'

def get_labeled_data():
    dset_list = list()
    with open(train, 'r', encoding="utf8") as file:
        #reader = csv.reader(file, delimiter='\t')
        reader=(line for line in csv.reader(file, delimiter = "\t", dialect='excel'))
        for row in reader:
            entry = dict()
            entry['text'] = row[1]

            entry['label'] = row[2]
        
            dset_list.append(entry)

    return dset_list

def get_data_binary():
    binary_data = get_labeled_data()
    for entry in binary_data:
        if entry["label"] == "Other":
            entry["label"] = "neutral"
        else:
            entry["label"] = "abusive"
    return binary_data