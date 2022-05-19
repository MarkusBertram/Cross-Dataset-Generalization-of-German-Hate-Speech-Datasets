import json
import pickle
import csv

data_path = "../data/HS_Dataset1/"

train = data_path + "germeval2018.training.txt"

test = data_path + "germeval2018.test.txt"

#unlabeled = data_path + "germeval2018.test.txt"

def read_file(filename):
    read_data=list()

    with open(filename, 'r', encoding='latin-1') as file:
        reader=csv.reader(file, delimiter='\t')

        for row in reader:
            entry=dict()
            entry['text'] = row[0]
            entry['label'] = row[1]
            entry['label2'] = row[2]
            read_data.append(entry)

    return read_data
    
    
    
def get_data():
    full_data = dict()    
    dset_list = list()    

    for dset in [train, test]:
        read_data = read_file(dset)
        dset_list.extend(read_data)

    full_data["labeled"] = dset_list

    #get unlabeled
    #read_data = read_file(unlabeled)
    #full_data["unlabeled"] = read_data

    return full_data