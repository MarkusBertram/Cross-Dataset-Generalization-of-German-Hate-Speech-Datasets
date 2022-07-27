import enum
from typing import Dict, Union, List, NoReturn
from abc import ABC, abstractmethod
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, BatchEncoding, Trainer, TrainingArguments, AdamW, get_scheduler
import re
import unicodedata
import sys
import numpy as np
from torchmetrics import F1Score

from utils.utils import (fetch_import_module, get_tweet_timestamp,
                         preprocess_text, print_data_example,
                         separate_text_by_classes)
import pandas as pd
#from datasets import Dataset, Features, ClassLabel, DatasetInfo
from torch.utils.data import Dataset, DataLoader, TensorDataset
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
        writer: SummaryWriter,
    ) -> NoReturn:
        #self.log_path = log_path
        self.writer = writer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained('deepset/gbert-base')
        self.basic_settings = basic_settings
        self.exp_settings = exp_settings
        self.abusive_ratio = None
        #basic_settings.update(exp_settings)
        #self.basic_settings = basic_settings

    @abstractmethod
    def train(self):
        pass

    # ovlossides test
    @torch.no_grad()
    def test(self):
        """test [computes loss of the test set]
        [extended_summary]
        Returns:
            [type]: [description]
        """
        correct = 0
        predictions = []
        targets = []
        f1 = F1Score(num_classes = 2, average="macro")
        self.model.eval()
        for (target_features, target_labels) in self.test_dataloader:
            target_features = target_features[0].to(self.device)
            target_labels = target_labels[0].int().to(self.device)

            target_class_output = self.model.inference(target_features)
            
            target_class_predictions = torch.round(torch.sigmoid(target_class_output)).int()

            predictions.append(target_class_predictions.cpu())
            targets.append(target_labels.cpu())
            
            correct += torch.sum(target_class_predictions == target_labels).item()

        avg_test_acc = correct / len(self.test_dataloader.dataset)

        outputs = torch.cat(predictions)
        targets = torch.cat(targets)
        f1score = f1(outputs, targets)

        self.writer.add_scalar(f"Accuracy/Test/{self.exp_name}", avg_test_acc)
        self.writer.add_scalar(f"F1_score/Test/{self.exp_name}", f1score.item())

        # add hparams
        self.writer.add_hparams(
            {
                "lr": self.lr,
                "epochs": self.epochs,
                "batchsize": self.batch_size,
                "train_split": self.train_split,
                "stratify_unlabelled": self.stratify_unlabelled

            },

            {
                "hparam/Accuracy/Test": avg_test_acc,
                "hparam/F1_score/Test": f1score.item()
            },
            run_name = self.exp_name
        )

    def get_target_dataset(self):

        # fetch labelled target dataset and split labelled target dataset into train and test
        labelled_target_dataset_features, labelled_target_dataset_labels = self.fetch_dataset(self.target_labelled, labelled = True, target = True)

        indices = np.arange(len(labelled_target_dataset_features))
        random_indices = np.random.permutation(indices)
        labelled_target_train_size = int(self.train_split * len(labelled_target_dataset_features))
        train_indices = random_indices[:labelled_target_train_size]
        test_indices = random_indices[labelled_target_train_size:]
        
        labelled_target_features_train = labelled_target_dataset_features[train_indices]
        labelled_target_features_test = labelled_target_dataset_features[test_indices]

        labelled_target_labels_train = labelled_target_dataset_labels[train_indices]
        labelled_target_labels_test = labelled_target_dataset_labels[test_indices]

        return labelled_target_features_train, labelled_target_labels_train, labelled_target_features_test, labelled_target_labels_test
    
    def preprocess(self, batch):
        batch = cleanTweets(batch)
        #batch["text"] = [cleanTweets(b) for b in batch["text"]]
        #batch = [cleanTweets(b) for b in batch]
        return pd.Series(self.tokenizer(batch, padding="max_length", truncation=True, max_length=512, return_token_type_ids = False))
    
    def fetch_dataset(self, dataset_name, labelled = True, target = False):

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
            dset_list_of_dicts = dset_module.get_data_binary(self.unlabelled_size, stratify = self.stratify_unlabelled, abusive_ratio = self.abusive_ratio)
        # convert list to dataframe
        dset_df = pd.DataFrame(dset_list_of_dicts)
        if labelled == True and target == True:
           self.abusive_ratio = dset_df["label"].value_counts(normalize = True)["abusive"]
        
        # tokenize each row in dataframe
        #tokens_df = dset_df.apply(lambda row: self.preprocess(row.text), axis='columns', result_type='expand')
        tokens_df = dset_df.apply(lambda row: self.preprocess(row.text), axis='columns', result_type='expand')
        tokens_array = np.array(tokens_df[["input_ids", "attention_mask"]].values.tolist())
        tokens_tensor = torch.from_numpy(tokens_array)

        if labelled == True:
            # map neutral to 0 and abusive to 1
            label_df = dset_df["label"].map(label2id)
            labels_array = np.array(label_df.values.tolist())
            labels_tensor = torch.from_numpy(labels_array).float()
        else:
            labels_tensor = None

        return tokens_tensor, labels_tensor


    def load_basic_settings(self):
        # data settings
        # self.labelled_size = self.basic_settings.get("labelled_size", 3000)
        self.target_labelled = self.basic_settings.get("target_labelled", "telegram_gold")
        self.target_unlabelled = self.basic_settings.get("target_unlabelled", "telegram_unlabeled")
        self.unlabelled_size = self.basic_settings.get("unlabelled_size", 200000)
        self.validation_split = self.basic_settings.get("validation_split", 0)
        self.train_split = self.basic_settings.get("train_split", 0.05)
        self.sources = self.basic_settings.get("sources", [
            "germeval2018", 
            "germeval2019",
            "hasoc2019",
            "hasoc2020",
            "ihs_labelled",
            "covid_2021"
        ])
        self.num_workers = self.basic_settings.get("num_workers", 8)
        # training settings
        self.freeze_BERT_weights = self.basic_settings.get("freeze_BERT_weights", True)
        self.stratify_unlabelled  = self.basic_settings.get("stratify_unlabelled", True)
        # training settings
        self.epochs = self.basic_settings.get("epochs", 100)
        self.batch_size = self.basic_settings.get("batch_size", 128)
        self.weight_decay = self.basic_settings.get("weight_decay", 1e-4)
        self.metric = self.basic_settings.get("metric", 10)
        self.lr = self.basic_settings.get("lr", 0.1)
        self.nesterov = self.basic_settings.get("nesterov", False)
        self.momentum = self.basic_settings.get("momentum", 0.9)
        self.lr_sheduler = self.basic_settings.get("lr_sheduler", True)
        self.num_classes = self.basic_settings.get("num_classes", 10)
        self.validation_split = self.basic_settings.get("validation_split", 0.3)
        self.validation_source = self.basic_settings.get(
            "validation_source", "test"
        )
        self.test_after_each_epoch = self.basic_settings.get("test_after_each_epoch", False)
        self.verbose = self.basic_settings.get("verbose", False)

        # self.criterion = self.basic_settings.get("criterion", "crossentropy")
        #self.metric = self.basic_settings.get("metric", "accuracy")
        
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