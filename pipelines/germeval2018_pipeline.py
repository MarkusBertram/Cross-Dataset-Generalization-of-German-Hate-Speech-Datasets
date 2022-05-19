import json
import pickle
import csv

data_path = "../data/HS_Dataset1/"

train = data_path + "germeval2018.training.txt"

test = data_path + "germeval2018.test.txt"

#unlabeled = data_path + "HS_Dataset1_unlabeled.txt"

def read_file(filename):
    read_data=list()

    with open(filename, 'r', encoding='latin-1') as file:
        reader=csv.reader(file, delimiter='\t')

        for row in reader:
            entry=dict()
            entry['text'] = row[0]
            entry['label_task1'] = row[1]
            entry['label_task2'] = row[2]
            read_data.append(entry)

    return read_data
    
    
    
def get_data():
    full_data = dict()
    
    # get train
    read_data = read_file(train)
    full_data["train"] = read_data

    # get test
    read_data = read_file(test)
    full_data["test"] = read_data

    ## get unlabeled
    #read_data = read_file(unlabeled)
    #full_data["unlabeled"] = read_data

    return full_data