from typing import Dict, List, Union

# python
import datetime
import os
import json
from xml.etree.ElementPath import prepare_descendant
import numpy as np
from src.model.MDAN_model import MDAN_model     
from src.utils.utils import (fetch_import_module, get_tweet_timestamp,
                         preprocess_text, print_data_example,
                         separate_text_by_classes)

import pandas as pd
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
import torch.nn.functional as F
from src.experiments.experiment_base import experiment_base
from src.utils.exp_utils import CustomConcatDataset

def multi_data_loader(source_dataloader_list, unlabelled_target_dataloader):

    # Custom Data Generator that yields source batches and an unlabelled target batch

    # Number of batches = number of batches of the larget dataset dataloader
    input_sizes = [len(dloader) for dloader in source_dataloader_list]
    number_of_batches = max(input_sizes)

    # create iter instances of each dataloader
    unlabelled_target_iter = iter(unlabelled_target_dataloader)
    src_dataloader_iter_list = []
    for dataloader in source_dataloader_list:
        src_dataloader_iter_list.append(iter(dataloader))

    # yield batches until the number of batches is reached
    for batch in range(number_of_batches):
        # iterate over source dataloader iters
        source_batches = []
        for i in range(len(src_dataloader_iter_list)):
            # append next iter
            try:
                source_batches.append(next(src_dataloader_iter_list[i]))
            # if dataloader is exhausted, reset dataloader iter
            except StopIteration:
                src_dataloader_iter_list[i] = iter(source_dataloader_list[i])
                source_batches.append(next(src_dataloader_iter_list[i]))
            
        try:
            unlabelled_target_batch = next(unlabelled_target_iter)
        except StopIteration:
            unlabelled_target_iter = iter(unlabelled_target_dataloader)
            unlabelled_target_batch = next(unlabelled_target_iter)

        yield source_batches, unlabelled_target_batch

class experiment_MDAN(experiment_base):
    def __init__(
        self,
        basic_settings: Dict,
        exp_settings: Dict,
        #log_path: str,
        writer: SummaryWriter,

    ):
        super(experiment_MDAN, self).__init__(basic_settings, exp_settings, writer)#, log_path, writer)
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
            # https://github.com/hanzhaoml/MDAN/blob/4eaec5b0d49a3af446b3b52850f646dfed507a14/main_amazon.py#L123
            self.model.train()

            train_dataloader = multi_data_loader(self.source_dataloader_list, self.unlabelled_target_dataloader)

            for i, (source_batches, unlabelled_target_batch) in enumerate(train_dataloader):
                self.optimizer.zero_grad()

                source_features = torch.stack([batch[0][0] for batch in source_batches]).to(self.device)
                source_labels = torch.stack([batch[1][0] for batch in source_batches]).to(self.device)
                
                unlabelled_target_features = unlabelled_target_batch[0][0].to(self.device)
                
                slabels = torch.ones(self.batch_size).type(torch.FloatTensor).to(self.device)
                tlabels = torch.zeros(self.batch_size).type(torch.FloatTensor).to(self.device)

                class_probabilities, source_domain_probabilities, target_domain_probabilities = self.model(source_features, unlabelled_target_features)

                # Compute prediction accuracy on multiple training sources.
                losses = torch.stack([self.bce_loss(class_probabilities[j], source_labels[j]) for j in range(len(self.source_dataloader_list))])

                # Compute domain discrepency loss on the training sources and the unlabelled target batch
                domain_losses = torch.stack([self.bce_loss(source_domain_probabilities[j], slabels) + self.bce_loss(target_domain_probabilities[j], tlabels) for j in range(len(self.source_dataloader_list))])

                # Soft version loss (6)
                loss = torch.log(torch.sum(torch.exp(self.gamma * (losses + self.mu * domain_losses)))) / self.gamma

                loss.backward()
                self.optimizer.step()
                
    def create_optimizer(self) -> None:
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas = (self.beta1, self.beta2))

    def create_criterion(self) -> None:
        self.bce_loss = nn.BCEWithLogitsLoss()

    # overrides load_settings
    def load_exp_settings(self) -> None:
        self.exp_name = self.current_experiment.get("exp_name", "MDAN_cnn")   
        self.gamma = self.current_experiment.get("gamma", 52)
        self.mu = self.current_experiment.get("mu", 0.002)
        self.lr = self.current_experiment.get("lr", 2.95e-5)
        self.beta1 = self.current_experiment.get("beta1", 0.92)
        self.beta2 = self.current_experiment.get("beta1", 0.993)

    def create_model(self):
        
        from src.model.feature_extractors import BERT_cnn
        #import .model.feature_extractors
        feature_extractor = BERT_cnn(self.bottleneck_dim)
        output_hidden_states = True

        from src.model.task_classifiers import DANN_task_classifier
        task_classifier = DANN_task_classifier(self.bottleneck_dim, self.layer_size)
       
        from src.model.domain_classifiers import DANN_domain_classifier
        domain_classifier = DANN_domain_classifier(self.bottleneck_dim, self.layer_size)
        
        self.model = MDAN_model(feature_extractor, task_classifier, domain_classifier, output_hidden_states, len(self.source_dataloader_list)).to(self.device)

    def create_dataloader(self):
        # fetch source datasets
        source_datasets = []
        for source_name in self.sources:
            train_features, val_features, train_labels, val_labels = self.fetch_dataset(source_name, labelled = True, target = False)
            source_datasets.append(TensorDataset(train_features, train_labels))
        
        labelled_target_features_train, labelled_target_labels_train, labelled_target_features_val, labelled_target_labels_val, labelled_target_features_test, labelled_target_labels_test = self.get_target_dataset()

        # add labelled target train dataset to source domains
        source_datasets.append(TensorDataset(labelled_target_features_train, labelled_target_labels_train))
        
        gc.collect()

        # create source dataloaders
        self.source_dataloader_list = []
        for source_dataset in source_datasets:
            sampler = BatchSampler(RandomSampler(source_dataset), batch_size=self.batch_size, drop_last=True)
            dataloader = DataLoader(dataset=source_dataset, sampler = sampler, num_workers=self.num_workers)            
            self.source_dataloader_list.append(dataloader)

        # fetch unlabelled target dataset and create dataloader
        unlabelled_target_dataset_features = self.fetch_dataset(self.target_unlabelled, labelled = False, target = True)
        unlabelled_target_dataset = TensorDataset(unlabelled_target_dataset_features)
        sampler = BatchSampler(RandomSampler(unlabelled_target_dataset), batch_size=self.batch_size, drop_last=True)
        self.unlabelled_target_dataloader = DataLoader(dataset=unlabelled_target_dataset, sampler = sampler, num_workers=self.num_workers)            
        
        

        # create test dataloader
        
        labelled_target_dataset_test = TensorDataset(labelled_target_features_test, labelled_target_labels_test)
        sampler = BatchSampler(RandomSampler(labelled_target_dataset_test), batch_size=self.batch_size, drop_last=False)
        self.test_dataloader = DataLoader(dataset=labelled_target_dataset_test, sampler = sampler, num_workers=self.num_workers)               
        del labelled_target_dataset_test
        gc.collect()

    # overrides perform_experiment
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
        
        # perform test
        self.test()