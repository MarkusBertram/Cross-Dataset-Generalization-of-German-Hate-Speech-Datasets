from typing import Dict, List, Union

# python
import datetime
import os
import json
import torch.nn.functional as F
from xml.etree.ElementPath import prepare_descendant
import numpy as np
from src.model.MME_model import MME_model     
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

def inv_lr_scheduler(optimizer, iter_num, gamma=0.0001,
                     power=0.75, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (- power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr## * param_lr[i]
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr * param_lr[i]
    #     i += 1
    return optimizer

class experiment_MME(experiment_base):
    def __init__(
        self,
        basic_settings: Dict,
        exp_settings: Dict,
        #log_path: str,
        writer: SummaryWriter,

    ):
        super(experiment_MME, self).__init__(basic_settings, exp_settings, writer)#, log_path, writer)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.current_experiment = exp_settings

        

    # overrides train
    def train(self):
        """train [main training function of the project]
        """

        for name, param in self.model.named_parameters():
            if "bert" in name:
                param.requires_grad = False

        data_iter_s = iter(self.source_dataloader)
        data_iter_t = iter(self.labelled_target_dataloader)
        data_iter_t_unl = iter(self.unlabelled_target_dataloader)
        len_train_source = len(self.source_dataloader)
        len_train_target = len(self.labelled_target_dataloader)
        len_train_target_semi = len(self.unlabelled_target_dataloader)

        steps = self.epochs*len(self.source_dataloader)

        for step in range(steps):

            if step % len_train_target == 0:
                data_iter_t = iter(self.labelled_target_dataloader)
            if step % len_train_target_semi == 0:
                data_iter_t_unl = iter(self.unlabelled_target_dataloader)
            if step % len_train_source == 0:
                data_iter_s = iter(self.source_dataloader)

            data_s = next(data_iter_s)
            data_t = next(data_iter_t)
            data_t_unl = next(data_iter_t_unl) 

            source_features = data_s[0][0].to(self.device)
            source_labels = data_s[1][0].to(self.device)
            labelled_target_features = data_t[0][0].to(self.device)
            labelled_target_labels = data_t[1][0].to(self.device)
            unlabelled_target_features = data_t_unl[0][0].to(self.device)
            
            self.optimizer_g.zero_grad()
            self.optimizer_f.zero_grad()

            data = torch.cat((source_features, labelled_target_features), 0)
            target = torch.cat((source_labels, labelled_target_labels), 0)

            output = self.model(data)

            loss = self.criterion(output, target)
            loss.backward(retain_graph=True)

            self.optimizer_g.step()
            self.optimizer_f.step()

            self.optimizer_g.zero_grad()
            self.optimizer_f.zero_grad()

            out_t1 = self.model(unlabelled_target_features, reverse = True, eta = self.eta)
            # conditional entropy
            # https://github.com/VisionLearningGroup/SSDA_MME/blob/81c3a9c321f24204db7223405662d4d16b22b17c/utils/loss.py#L36
            loss_t = self.lamda * torch.mean(torch.sigmoid(out_t1)*F.logsigmoid(out_t1+1e-5))

            loss_t.backward()
            self.optimizer_f.step()
            self.optimizer_g.step()


    def create_optimizer(self) -> None:
        params = [{
            "params": self.model.feature_extractor.parameters(), "lr": self.lr
        }]
        self.optimizer_g = optim.Adam(
            params,
            lr=self.lr,
            betas = (self.beta1, self.beta2)
        )
        params = [{
            "params": self.model.task_classifier.parameters(), "lr": self.lr
        }]

        self.optimizer_f = optim.Adam(
            params,
            lr=self.lr,
            betas = (self.beta1, self.beta2)
        )


    def create_criterion(self) -> None:
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)

    # overrides load_settings
    def load_exp_settings(self) -> None:
        self.exp_name = self.current_experiment.get("exp_name", "MME_cnn")   
        self.lamda = self.current_experiment.get("lamda", 0.014)
        self.eta = self.current_experiment.get("eta", 0.36)
        self.lr = self.current_experiment.get("lr", 8.89e-5)
        self.beta1 = self.current_experiment.get("beta1", 0.888)
        self.beta2 = self.current_experiment.get("beta1", 0.999)

    def create_model(self):
        
        from src.model.feature_extractors import BERT_cnn
        feature_extractor = BERT_cnn(self.bottleneck_dim)
        
        from src.model.task_classifiers import DANN_task_classifier
        task_classifier = DANN_task_classifier(self.bottleneck_dim, self.layer_size)

        self.model = MME_model(feature_extractor, task_classifier).to(self.device)

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
        
        # concatenate datasets
        combined_source_features = torch.cat(source_features)
        combined_source_labels = torch.cat(source_labels)
        
        # create source dataloader
        source_dataset = TensorDataset(combined_source_features, combined_source_labels)
        sampler = BatchSampler(RandomSampler(source_dataset), batch_size=self.batch_size, drop_last=True)
        self.source_dataloader = DataLoader(dataset=source_dataset, sampler = sampler, num_workers=self.num_workers)            
        
        del source_features
        del source_labels
        gc.collect()

        # create labelled target dataloader
        labelled_target_dataset_train = TensorDataset(labelled_target_features_train, labelled_target_labels_train)

        sampler = BatchSampler(RandomSampler(labelled_target_dataset_train), batch_size=min(self.batch_size, len(labelled_target_dataset_train)), drop_last=True)
        self.labelled_target_dataloader = DataLoader(dataset=labelled_target_dataset_train, sampler = sampler, num_workers=self.num_workers)            

        # fetch unlabelled target dataset and create dataloader
        unlabelled_target_dataset_features = self.fetch_dataset(self.target_unlabelled, labelled = False, target = True)
        unlabelled_target_dataset = TensorDataset(unlabelled_target_dataset_features)
        sampler = BatchSampler(RandomSampler(unlabelled_target_dataset), batch_size=2 * self.batch_size, drop_last=True)
        self.unlabelled_target_dataloader = DataLoader(dataset=unlabelled_target_dataset, sampler = sampler, num_workers=self.num_workers)            
        
        

        # create test dataloader
        labelled_target_dataset_test = TensorDataset(labelled_target_features_test, labelled_target_labels_test)
        sampler = BatchSampler(RandomSampler(labelled_target_dataset_test), batch_size=2*self.batch_size, drop_last=True)
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