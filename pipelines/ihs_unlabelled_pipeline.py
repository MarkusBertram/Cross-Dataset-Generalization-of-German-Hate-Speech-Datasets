from pathlib import Path
import re
import random

data_path = Path(__file__).resolve().parents[1] / 'data' / 'ihs_unlabelled'

train = data_path / 'tweets.csv'

r = re.compile(r'[0-9]{19},')

def get_labeled_data(unlabelled_size):
    dset_list = []

    with open(train, 'r', encoding="utf-8") as file:
        for line in file:
            try:
                entry = {}
                split = re.split(r, line)
                if len(split) == 2:
                    entry['text'] = split[1].replace("\n", '')
                
                    dset_list.append(entry)
            except:
                pass

    # select random elements from list to return
    dset_list = random.choices(dset_list, k=unlabelled_size)

    return dset_list

def get_data_binary(unlabelled_size, stratify, abusive_ratio):
    return get_labeled_data(unlabelled_size)