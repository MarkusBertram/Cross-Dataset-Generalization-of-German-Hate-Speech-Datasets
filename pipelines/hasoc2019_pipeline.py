import json
import pickle
import csv
from pathlib import Path

data_path = Path(__file__).resolve().parents[1] / 'data' / 'hasoc2019'

train = data_path / "german_dataset.tsv"

test = data_path / "hasoc_de_test_gold.tsv"

#unlabeled = data_path + "germeval2018.test.txt"

def read_file(filename):
    read_data=list()

    with open(filename, 'r', encoding="utf8") as file:
        reader=csv.reader(file, delimiter='\t')

        for row in reader:
            entry=dict()
            entry['text'] = row[1]
            entry['label'] = row[2]
            entry['fine-grained_label'] = row[3]
            read_data.append(entry)

    return read_data[1:]
    
    
def get_labeled_data():  
    dset_list = list()     

    for dset in [train, test]:
        read_data = read_file(dset)
        dset_list.extend(read_data)


    #get unlabeled
    #read_data = read_file(unlabeled)
    #full_data["unlabeled"] = read_data

    return dset_list