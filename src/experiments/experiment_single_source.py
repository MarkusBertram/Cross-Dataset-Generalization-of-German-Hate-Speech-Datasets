from typing import Dict, List, Union

# python
import datetime
import os
import json
from xml.etree.ElementPath import prepare_descendant
import numpy as np
from src.model.labelled_only_model import labelled_only_model
from src.utils.utils import (fetch_import_module, get_tweet_timestamp,
                         preprocess_text, print_data_example,
                         separate_text_by_classes)

import pandas as pd
from pathlib import Path
from torchmetrics import F1Score
import yaml
from torch.utils.tensorboard.writer import SummaryWriter
from sklearn.model_selection import train_test_split
import sys
from time import time
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
#from torchsummary import summary
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, BatchEncoding, Trainer, TrainingArguments, AdamW, get_scheduler
import gc
from torch.utils.data.dataset import ConcatDataset
#from .helpers.measures import accuracy, auroc, f1
from torch.utils.data import Dataset, DataLoader, TensorDataset

from src.experiments.experiment_base import experiment_base
from src.utils.exp_utils import CustomConcatDataset

class experiment_single_source(experiment_base):
    def __init__(
        self,
        basic_settings: Dict,
        exp_settings: Dict,
        writer: SummaryWriter,

    ):
        super(experiment_single_source, self).__init__(basic_settings, exp_settings, writer)#, log_path, writer)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.current_experiment = exp_settings

    # overrides train
    def train(self):
        """train [main training function of the project]
        [extended_summary]
        Args:
            train_loader ([torch.Dataloader]): [dataloader with the training data]
            optimizer ([torch.optim]): [optimizer for the network]
            criterion ([Loss function]): [Pytorch loss function]
            device ([str]): [device to train on cpu/cuda]
            epochs (int, optional): [epochs to run]. Defaults to 5.
            **kwargs (verbose and validation dataloader)
        Returns:
            [tupel(trained network, train_loss )]:
        """

        for name, param in self.model.named_parameters():
            if "bert" in name:
                param.requires_grad = False

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0

            for i, (target_features, target_labels) in enumerate(self.train_dataloader):

                self.optimizer.zero_grad(set_to_none=True)

                # training model
                target_features = target_features[0].to(self.device)
                target_labels = target_labels[0].to(self.device)
                
                class_output = self.model(input_data=target_features)
                
                loss = self.criterion(class_output, target_labels)

                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            # self.writer.add_scalar(f"total_loss/train/{self.exp_name}", total_loss, epoch)

    def create_optimizer(self) -> None:
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas = (self.beta1, self.beta2)
        )

    def create_criterion(self) -> None:
        self.criterion = nn.BCEWithLogitsLoss()

    # overrides load_settings
    def load_exp_settings(self) -> None:
        self.source_name = self.current_experiment.get("source_name", "germeval2018")
        self.targets = self.current_experiment.get("targets", [
              "germeval2018", 
              "germeval2019",
              "hasoc2019",
              "hasoc2020",
              "ihs_labelled",
              "covid_2021"
          ])
        self.fair = self.current_experiment.get("fair", False)
        
        self.lr = self.current_experiment.get("lr", 5.2e-5)
        self.beta1 = self.current_experiment.get("beta1", 0.875)
        self.beta2 = self.current_experiment.get("beta2", 0.945)
        self.save_model = self.current_experiment.get("save_model", False)

    def create_model(self):
        
        from src.model.feature_extractors import BERT_cnn
        feature_extractor = BERT_cnn(self.bottleneck_dim)

        from src.model.task_classifiers import DANN_task_classifier
        task_classifier = DANN_task_classifier(self.bottleneck_dim, self.layer_size)

        self.model = labelled_only_model(feature_extractor, task_classifier).to(self.device)

    def create_dataloader(self):

        ####### fetch source dataset

        # fetch datasets
        label2id = {"neutral": 0, "abusive":1}
        # import dataset pipeline
        source_module = fetch_import_module(self.source_name)
        # execute get_data_binary in pipeline
        dset_list_of_dicts = source_module.get_data_binary()
        # convert list to dataframe
        dset_df = pd.DataFrame(dset_list_of_dicts)

        # tokenize each row in dataframe
        tokens_df = dset_df.apply(lambda row: self.preprocess(row.text), axis='columns', result_type='expand')

        source_tokens_array = np.array(tokens_df[["input_ids", "attention_mask"]].values.tolist())
        #map neutral to 0 and abusive to 1
        source_label_df = dset_df["label"].map(label2id)
        source_labels_array = np.array(source_label_df.values.tolist())

        # "fair" is the scenario where all datasets have the same training and test size
        # "unfair" is the scenario where 80% of each total dataset is used for train, 20% for test
        if self.fair == True:
            train_size = 919
            test_size = 230
        else:
            train_size = 0.8
            test_size = 0.2

        source_features_train, _, source_labels_train, _ = train_test_split(source_tokens_array, source_labels_array, train_size = train_size, test_size = test_size, random_state=self.seed, stratify = source_labels_array)

        train_dataset = TensorDataset(torch.from_numpy(source_features_train), torch.from_numpy(source_labels_train).float())

        # create train dataloader
        sampler = BatchSampler(RandomSampler(train_dataset), batch_size=self.batch_size, drop_last=False)
        self.train_dataloader = DataLoader(dataset=train_dataset, sampler = sampler, num_workers=self.num_workers)                        

    def create_test_dataloader(self):
        ####### fetch source dataset

        # fetch datasets
        label2id = {"neutral": 0, "abusive":1}
        # import dataset pipeline
        source_module = fetch_import_module(self.source_name)
        # execute get_data_binary in pipeline
        dset_list_of_dicts = source_module.get_data_binary()
        # convert list to dataframe
        dset_df = pd.DataFrame(dset_list_of_dicts)

        # tokenize each row in dataframe
        tokens_df = dset_df.apply(lambda row: self.preprocess(row.text), axis='columns', result_type='expand')

        source_tokens_array = np.array(tokens_df[["input_ids", "attention_mask"]].values.tolist())
        #map neutral to 0 and abusive to 1
        source_label_df = dset_df["label"].map(label2id)
        source_labels_array = np.array(source_label_df.values.tolist())

        # "fair" is the scenario where all datasets have the same training and test size
        # "unfair" is the scenario where 80% of each total dataset is used for train, 20% for test
        if self.fair == True:
            train_size = 919
            test_size = 230
        else:
            train_size = 0.8
            test_size = 0.2

        _, source_features_test, _, source_labels_test = train_test_split(source_tokens_array, source_labels_array, train_size = train_size, test_size = test_size, random_state=self.seed, stratify = source_labels_array)

        if self.source_name == self.target_name:
            test_dataset = TensorDataset(torch.from_numpy(source_features_test), torch.from_numpy(source_labels_test).float())
        else:
            # import dataset pipeline
            target_module = fetch_import_module(self.target_name)
            # execute get_data_binary in pipeline
            target_dset_list_of_dicts = target_module.get_data_binary()
            # convert list to dataframe
            target_dset_df = pd.DataFrame(target_dset_list_of_dicts)

            # tokenize each row in dataframe
            target_tokens_df = target_dset_df.apply(lambda row: self.preprocess(row.text), axis='columns', result_type='expand')

            target_tokens_array = np.array(target_tokens_df[["input_ids", "attention_mask"]].values.tolist())
            #map neutral to 0 and abusive to 1
            target_label_df = target_dset_df["label"].map(label2id)
            target_labels_array = np.array(target_label_df.values.tolist())
            
            _, target_features_test, _, target_labels_test = train_test_split(target_tokens_array, target_labels_array, train_size=train_size, test_size = test_size, random_state=self.seed, stratify = target_labels_array)

            test_dataset = TensorDataset(torch.from_numpy(target_features_test), torch.from_numpy(target_labels_test).float())
        
        # create test dataloader
        sampler = BatchSampler(RandomSampler(test_dataset), batch_size=self.batch_size, drop_last=False)
        self.test_dataloader = DataLoader(dataset=test_dataset, sampler = sampler, num_workers=self.num_workers) 

    # ovlossides perform_experiment
    def perform_experiment(self):
        # load basic settings
        self.load_basic_settings()

        # load specific experiment settings
        self.load_exp_settings()

        # fetch datasets and create dataloader
        self.create_dataloader()

        # create model using components
        self.create_model()

        # create optimizer
        self.create_optimizer()

        # create criterion
        self.create_criterion()

        # perform train
        self.train()

        for t_name in self.targets:
            self.target_name = t_name
            self.exp_name = f"single_source_{self.source_name}_{self.target_name}_fair_{str(self.fair)}"
            
            # fetch datasets and create dataloader
            self.create_test_dataloader()

            # perform test
            self.test()

            if self.source_name == self.target_name and self.save_model == True:
                path_state_dict = Path(f"./model_state_dicts/{self.exp_name}")
                path_state_dict.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), str(path_state_dict)+f"/{self.exp_name}_model.pt")

