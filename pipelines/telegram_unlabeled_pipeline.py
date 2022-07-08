import bson
import sys
from pathlib import Path
from itertools import islice
import math
from collections import Counter
data_path = Path(__file__).resolve().parents[1] / 'data' / 'telegram_unlabelled'

train = data_path / 'messages.bson'

def map_values(list):
    key_map = {"neutral": ["OTHER", "NOT", "not", "NONE"],
                "abusive": ["OFFENSE", "INSULT", "PROFANITY", "ABUSE", "HATE", "PRFN", "OFFN", "HOF"]
    }
    for i in range(len(list)):
        if list[i] in key_map["neutral"]:
            list[i] = "neutral"
        elif list[i] in key_map["abusive"]:
            list[i] = "abusive"
    return list

def get_labeled_data(unlabelled_size, stratify, abusive_ratio):
    dset_list = []
    
    with open(train, 'rb') as f:
        data = bson.decode_file_iter(f)
        if stratify == False:
            count = 0
            for d in data:
                if "language" in d:
                    if d["language"] == "de":
                        entry = dict()
                        entry["text"] = d["text"]
                        dset_list.append(entry)
                        count += 1
                if count == unlabelled_size:
                    return dset_list
        else:
            neutral_size = int(unlabelled_size * (1-abusive_ratio))
            abusive_size = int(unlabelled_size * (abusive_ratio))
            count_neutral = 0
            count_abusive = 0
            classifiers = ('hate_germeval_1819_2', 'hate_germeval_18', 'hate_germeval_19', 'hate_germeval_1819', 'hate_germeval_1819_task_2', 'hate_germeval_18_task_2', 'hate_germeval_19_task_2', 'hate_hasoc_2020', 'hate_hasoc_1920', 'hate_covid_2021', 'hate_hasoc_1920_task_2', 'hate_hasoc_2020_task_2' )
            for d in data:
                if "language" in d:
                    if d["language"] == "de":
            
                        labels = [d[key]["label"] for key in classifiers if key in d]
                        counter = Counter(map_values(labels))
                        label = counter.most_common()[0][0]
                        most_common_count = counter.most_common()[0][1]

                        if most_common_count >= int(len(labels)/2)+1:
                            if label == "neutral" and count_neutral < neutral_size:
                                entry = dict()
                                entry['text'] = d["text"]
                                entry['label'] = 'neutral'
                                dset_list.append(entry)
                                count_neutral += 1
                            elif label == "abusive" and count_abusive < abusive_size:
                                entry = dict()
                                entry['text'] = d["text"]
                                entry['label'] = 'abusive'
                                dset_list.append(entry)
                                count_abusive += 1
                if count_neutral == neutral_size and count_abusive == abusive_size:
                    return dset_list

def get_data_binary(unlabelled_size, stratify, abusive_ratio):
    return get_labeled_data(unlabelled_size, stratify, abusive_ratio)