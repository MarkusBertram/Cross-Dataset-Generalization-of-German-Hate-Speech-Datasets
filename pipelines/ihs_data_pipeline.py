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
    return dset_list

get_labeled_data()