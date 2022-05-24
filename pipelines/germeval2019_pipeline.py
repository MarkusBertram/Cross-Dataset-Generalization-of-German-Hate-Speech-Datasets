import json
import pickle
import csv
from pathlib import Path

data_path = Path(__file__).resolve().parents[1] / 'data' / 'germeval2019'

train1 = data_path / "germeval2019.training_subtask1_2_korrigiert.txt"
train2 = data_path / "germeval2019.training_subtask3.txt"

test1 = data_path / "germeval2019GoldLabelsSubtask1_2.txt"
test2 = data_path / "germeval2019_Testdata_Subtask3.txt"

def read_file(filename):
    read_data=list()

    with open(filename, 'r', encoding="utf8") as file:
        reader=csv.reader(file, delimiter='\t')

        for row in reader:
            if len(row) > 0:
                entry=dict()
                entry['text'] = row[0]
                entry['label'] = row[1]
                entry['fine-grained_label'] = row[2]
                read_data.append(entry)

    return read_data
    
    
    
def get_labeled_data(): 
    dset_list = list()    

    for dset in [train1, train2, test1, test2]:
        read_data = read_file(dset)
        dset_list.extend(read_data)

    #get unlabeled
    #read_data = read_file(unlabeled)
    #full_data["unlabeled"] = read_data

    return dset_list