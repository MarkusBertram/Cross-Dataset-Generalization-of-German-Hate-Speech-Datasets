import enum
from typing import Dict, Union, List, NoReturn
from abc import ABC, abstractmethod
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, BatchEncoding, Trainer, TrainingArguments, AdamW, get_scheduler
import re
import unicodedata
import sys
from utils.utils import (fetch_import_module, get_tweet_timestamp,
                         preprocess_text, print_data_example,
                         separate_text_by_classes)
import pandas as pd
from datasets import Dataset, Features, ClassLabel, DatasetInfo

twitter_username_re = re.compile(r'@([A-Za-z0-9_]+)')
hashtag_re = re.compile(r'\B(\#[a-zA-Z0-9]+\b)(?!;)')
html_symbol_re = re.compile(r'&[^ ]+')
lbr_re = re.compile(r'\|LBR\|')
url_re = re.compile(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})')

def cleanTweets(tweet):
    text = twitter_username_re.sub("[UNK]",tweet)
    text = lbr_re.sub(' ', text)
    text = unicodedata.normalize('NFKC',text)
    text = text.replace('\n',' ')
    text = text.replace('RT ',' ')
    text = hashtag_re.sub("[UNK]",text)
    text = html_symbol_re.sub(" ",text)
    text = url_re.sub("[UNK]",text)
    return text

class experiment_base(ABC):
    """experiment_base [abstract base class other experiments ]"""

    def __init__(
        self,
        basic_settings: Dict,
        exp_settings: Dict,
        #log_path: str,
        #writer: SummaryWriter,
    ) -> NoReturn:
        #self.log_path = log_path
        #self.writer = writer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained('deepset/gbert-base')
        self.basic_settings = basic_settings
        self.exp_settings = exp_settings
        self.abusive_ratio = None
        #basic_settings.update(exp_settings)
        #self.current_experiment = basic_settings

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    def preprocess(self, batch):
        
        batch["text"] = [cleanTweets(b) for b in batch["text"]]

        return self.tokenizer(batch["text"], padding="longest", truncation=True, max_length=512, return_tensors="pt")
    
    def fetch_dataset(self, dataset_name, labelled = True, target = False):

        dsetinfo = DatasetInfo(description=dataset_name)
        ###### fetch datasets
        label2id = {"neutral": 0, "abusive":1}
        # import dataset pipeline
        dset_module = fetch_import_module(dataset_name)
        # execute get_data_binary in pipeline
        if labelled == True and target == False:
            dset_list_of_dicts = dset_module.get_data_binary()
        elif labelled == True and target == True:
            dset_list_of_dicts = dset_module.get_data_binary()
        elif labelled == False and target == True:
            dset_list_of_dicts = dset_module.get_data_binary(self.basic_settings["unlabelled_size"], stratify = self.basic_settings["stratify_unlabelled"], abusive_ratio = self.abusive_ratio)
        # convert list to dataframe
        dset_df = pd.DataFrame(dset_list_of_dicts)
        if labelled == True and target == True:
           self.abusive_ratio = dset_df["label"].value_counts(normalize = True)["abusive"]
        # convert dataframe to HuggingFace Dataset format
        dset_dataset = Dataset.from_pandas(dset_df, info=dsetinfo)
        dset_dataset = dset_dataset.class_encode_column("label")
        dset_dataset = dset_dataset.align_labels_with_mapping(label2id, "label")
        dset_dataset = dset_dataset.with_format("torch", device = self.device)
        # set lazy tokenize transform on the fly
        dset_dataset.set_transform(self.preprocess, columns = ["text"], output_all_columns = True)

        return dset_dataset


    def load_basic_settings(self):
        # data settings
        # self.labelled_size = self.current_experiment.get("labelled_size", 3000)
        self.target_labelled = self.current_experiment.get("target_labelled", "telegram_gold")
        self.target_unlabelled = self.current_experiment.get("target_unlabelled", "telegram_unlabeled")
        self.unlabelled_size = self.current_experiment.get("unlabelled_size", 200000)
        self.validation_split = self.current_experiment.get("validation_split", 0)
        self.test_split = self.current_experiment.get("test_split", 0.95)
        self.sources = self.current_experiment.get("sources", [
            "germeval2018", 
            "germeval2019",
            "hasoc2019",
            "hasoc2020",
            "ihs_labelled",
            "covid_2021"
        ])
        self.num_workers = self.current_experiment.get("num_workers", 8)
        # training settings
        self.freeze_BERT_weights = self.current_experiment.get("freeze_BERT_weights", True)

        # training settings
        self.epochs = self.current_experiment.get("epochs", 100)
        self.batch_size = self.current_experiment.get("batch_size", 128)
        self.weight_decay = self.current_experiment.get("weight_decay", 1e-4)
        self.metric = self.current_experiment.get("metric", 10)
        self.lr = self.current_experiment.get("lr", 0.1)
        self.nesterov = self.current_experiment.get("nesterov", False)
        self.momentum = self.current_experiment.get("momentum", 0.9)
        self.lr_sheduler = self.current_experiment.get("lr_sheduler", True)
        self.num_classes = self.current_experiment.get("num_classes", 10)
        self.validation_split = self.current_experiment.get("validation_split", 0.3)
        self.validation_source = self.current_experiment.get(
            "validation_source", "test"
        )
        # self.criterion = self.current_experiment.get("criterion", "crossentropy")
        self.create_criterion()
        self.metric = self.current_experiment.get("metric", "accuracy")


        
        #self.set_sampler(self.oracle)
        pass
    
    @abstractmethod
    def load_exp_settings(self) -> NoReturn:
        pass

    @abstractmethod
    def create_dataloader(self) -> NoReturn:
        pass

    @abstractmethod
    def create_model(self) -> NoReturn:
        pass

    # @abstractmethod
    # def create_plots(self) -> NoReturn:
    #     pass

    @abstractmethod
    def perform_experiment(self):
        pass