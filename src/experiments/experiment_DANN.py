from typing import Dict, List, Union

# python
import datetime
import os
import json
from xml.etree.ElementPath import prepare_descendant
import numpy as np
from src.model.DANN_model import DANN_model     
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

from src.experiments.experiment_base import experiment_base
from src.utils.exp_utils import CustomConcatDataset

class experiment_DANN(experiment_base):
    def __init__(
        self,
        basic_settings: Dict,
        exp_settings: Dict,
        #log_path: str,
        writer: SummaryWriter,

    ):
        super(experiment_DANN, self).__init__(basic_settings, exp_settings, writer)#, log_path, writer)
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

            len_dataloader = len(self.train_dataloader)

            for i, (source_batch, unlabelled_target_features) in enumerate(self.train_dataloader):

                self.optimizer.zero_grad(set_to_none=True)
                p = float(i + epoch * len_dataloader) / self.epochs / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                # training model using source data
                source_features = source_batch[0][0].to(self.device)
                source_labels = source_batch[1][0].to(self.device)
                
                #batch_size = len(source_labels)

                domain_label = torch.zeros_like(source_labels).to(self.device)
                #domain_label = domain_label.float()

                class_output, domain_output = self.model(input_data=source_features, alpha=alpha)
                
                loss_s_label = self.loss_class(class_output, source_labels)
                loss_s_domain = self.loss_domain(domain_output, domain_label)

                # training model using target data
                unlabelled_target_features = unlabelled_target_features[0].to(self.device)

                #batch_size = len(unlabelled_target_features)

                domain_label = torch.ones_like(source_labels).to(self.device)
                #domain_label = domain_label.float().to(self.device)

                _, domain_output = self.model(input_data=unlabelled_target_features, alpha=alpha)
                loss_t_domain = self.loss_domain(domain_output, domain_label)
                loss = loss_t_domain + loss_s_domain + loss_s_label

                total_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            self.writer.add_scalar(f"total_loss/train/{self.exp_name}", total_loss, epoch)

    def create_optimizer(self) -> None:
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas = (self.beta1, self.beta2)
        )

    def create_criterion(self) -> None:
        self.loss_class = nn.BCEWithLogitsLoss().to(self.device)
        self.loss_domain = nn.BCEWithLogitsLoss().to(self.device)

    # overrides load_settings
    def load_exp_settings(self) -> None:
        self.exp_name = self.current_experiment.get("exp_name", "DANN")
        self.lr = self.current_experiment.get("lr", 6.7e-5)
        self.beta1 = self.current_experiment.get("beta1", 0.911)
        self.beta2 = self.current_experiment.get("beta2", 0.998)
        self.gamma = self.current_experiment.get("gamma", 10)
        
    def create_model(self):
        
        from src.model.feature_extractors import BERT_cnn
        #import .model.feature_extractors
        feature_extractor = BERT_cnn(self.bottleneck_dim)
        output_hidden_states = True
        
        from src.model.task_classifiers import DANN_task_classifier
        task_classifier = DANN_task_classifier(self.bottleneck_dim, self.layer_size)
            
        from src.model.domain_classifiers import DANN_domain_classifier
        domain_classifier = DANN_domain_classifier(self.bottleneck_dim, self.layer_size)

        self.model = DANN_model(feature_extractor, task_classifier, domain_classifier, output_hidden_states).to(self.device)

    def create_dataloader(self):
        # fetch source datasets
        source_features = []
        source_labels = []

        self.unlabelled_size = 0
        for source_name in self.sources:
            train_features, val_features, train_labels, val_labels = self.fetch_dataset(source_name, labelled = True, target = False)
            self.unlabelled_size += len(train_features) + len(val_features)
            # discard validation features and labels
            source_features.append(train_features)
            source_labels.append(train_labels)
        
        # fetch labelled target dataset
        labelled_target_features_train, labelled_target_labels_train, labelled_target_features_val, labelled_target_labels_val, labelled_target_features_test, labelled_target_labels_test = self.get_target_dataset()
        
        source_features.append(labelled_target_features_train)
        source_labels.append(labelled_target_labels_train)

        combined_source_features = torch.cat(source_features)
        combined_source_labels = torch.cat(source_labels)
        
        # concatenate datasets
        source_dataset = TensorDataset(combined_source_features, combined_source_labels)
        
        del source_features
        del source_labels
        gc.collect()

        # fetch unlabelled target dataset
        
        unlabelled_target_dataset_features = self.fetch_dataset(self.target_unlabelled, labelled = False, target = True)
        
        # combine source dataset and unlabelled target dataset into one dataset
        concatenated_train_dataset = CustomConcatDataset(source_dataset, unlabelled_target_dataset_features)

        sampler = BatchSampler(RandomSampler(concatenated_train_dataset), batch_size=self.batch_size, drop_last=False)
        self.train_dataloader = DataLoader(dataset=concatenated_train_dataset, sampler = sampler, num_workers=self.num_workers)            
        #self.train_dataloader = DataLoader(concatenated_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        del concatenated_train_dataset
        
        gc.collect()

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