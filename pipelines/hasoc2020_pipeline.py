import json
import pickle
import csv
from pathlib import Path
import pandas as pd

data_path = Path(__file__).resolve().parents[1] / 'data' / 'hasoc2020'

train = data_path / "hasoc_2020_de_train_new.xlsx"

test = data_path / "hasoc_2020_de_test_new.xlsx"

#unlabeled = data_path + "germeval2018.test.txt"

def read_file(filename):
    read_data=list()

    excel_file = pd.read_excel(filename, header = 0, names = ['text', 'label', 'fine-grained_label'], usecols = "B,C,D", dtype={'text': str, 'label': str, 'fine-grained_label': str})

    return excel_file.to_dict('records')
    
    
def get_labeled_data():  
    dset_list = list()     

    for dset in [train, test]:
        read_data = read_file(dset)
        dset_list.extend(read_data)


    #get unlabeled
    #read_data = read_file(unlabeled)
    #full_data["unlabeled"] = read_data

    return dset_list

def get_data_binary():
    binary_data = get_labeled_data()
    for entry in binary_data:
        del entry["fine-grained_label"]
        if entry["label"] == "HOF":
            entry["label"] = "abusive"
        else:
            entry["label"] = "neutral"
    return binary_data