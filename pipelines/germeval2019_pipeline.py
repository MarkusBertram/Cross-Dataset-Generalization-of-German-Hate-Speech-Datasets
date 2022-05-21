import json
import pickle
import csv
from pathlib import Path

parent_folder = Path(__file__).resolve().parents[1]

data_path = parent_folder / 'data' / 'germeval2019'

train1 = data_path / "germeval2019.training_subtask1_2_korrigiert.txt"
train2 = data_path / "germeval2019.training_subtask3.txt"

test1 = data_path / "germeval2019GoldLabelsSubtask1_2.txt"
test2 = data_path / "germeval2019_Testdata_Subtask3.txt"

def read_file(filename):
    read_data=list()

    with open(filename, 'r', encoding='latin-1') as file:
        reader=csv.reader(file, delimiter='\t')

        for row in reader:
            if len(row) > 0:
                entry=dict()
                entry['text'] = row[0]
                entry['binary_label'] = row[1]
                entry['fine-grained_label'] = row[2]
                read_data.append(entry)

    return read_data
    
    
    
def get_data():
    full_data = dict()    
    dset_list = list()    

    for dset in [train1, train2, test1, test2]:
        read_data = read_file(dset)
        dset_list.extend(read_data)

    full_data["labeled"] = dset_list

    #get unlabeled
    #read_data = read_file(unlabeled)
    #full_data["unlabeled"] = read_data

    return full_data