import csv
import pickle
from pathlib import Path

data_path = Path(__file__).resolve().parents[1] / 'data' / 'Davidson_2017' / 'labeled_data.csv'

def get_labeled_data():
    dset_list = list()
    with open(data_path, 'r') as file:
        #reader = csv.reader(file, delimiter='\t')
        reader=(line for line in csv.reader(file, dialect='excel'))
        for row in reader:
            # if len(row) != 7:
            #     print(len(row))
            entry = dict()
            entry['text'] = row[-1]
            try:
                if int(row[-2]) == 0:
                    entry['label'] = 'hate'
                if int(row[-2]) == 1:
                    entry['label'] = 'offensive'
                if int(row[-2]) == 2:
                    entry['label'] = 'neither'
            except:
                entry['label'] = row[-2]
            dset_list.append(entry)

    return dset_list[1:]

def get_data_binary():
    dset_list = list()
    with open(data_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            entry = dict()
            entry['text'] = row[-1]
            try:
                if int(row[-2]) == 0:
                    entry['label'] = 'abusive'
                if int(row[-2]) == 1:
                    entry['label'] = 'abusive'
                if int(row[-2]) == 2:
                    entry['label'] = 'neutral'
            except:
                entry['label'] = row[-2]
            dset_list.append(entry)
    return dset_list[1:]