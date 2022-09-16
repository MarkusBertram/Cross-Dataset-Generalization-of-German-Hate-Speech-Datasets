from typing import Dict, List, Union

# python
import datetime
import os
import json
from xml.etree.ElementPath import prepare_descendant
import numpy as np
from src.model.M3SDA_model import M3SDA_model     
from src.utils.utils import (fetch_import_module, get_tweet_timestamp,
                         preprocess_text, print_data_example,
                         separate_text_by_classes)
from collections import defaultdict
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

class experiment_M3SDA(experiment_base):
    def __init__(
        self,
        basic_settings: Dict,
        exp_settings: Dict,
        #log_path: str,
        writer: SummaryWriter,

    ):
        super(experiment_M3SDA, self).__init__(basic_settings, exp_settings, writer)#, log_path, writer)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.current_experiment = exp_settings

        

    # overrides train
    def train(self):
        """train [main training function of the project]
        """

        N = len(self.source_dataloader_list)

        for name, param in self.model.named_parameters():
            if "bert" in name:
                param.requires_grad = False
        
        for epoch in range(1, self.epochs + 1):
            # https://github.com/tsxce/Moment-Matching-for-Multi-Source-Domain-Adaptation-M3SDA/blob/9f70e69113fefca0a6da30c91ebcf94f0cae682f/train.py#L86
            
            self.model.train()

            source_ac = {}
            for i in range(N):
                source_ac[i] = defaultdict(int)

            train_dataloader = multi_data_loader(self.source_dataloader_list, self.unlabelled_target_dataloader)

            for batch_index, (source_batches, unlabelled_target_batch) in enumerate(train_dataloader):

                loss_cls = 0
                
                ######### train extractor and source classifier, Step i.
                # First term of equation (2)
                for index, batch in enumerate(source_batches):
                    features = batch[0][0]
                    labels = batch[1][0]
                    
                    features = features.to(self.device)
                    labels = labels.to(self.device)

                    pred1, pred2 = self.model(features, index)

                    source_ac[index]['c1'] += torch.sum(torch.round(pred1) == labels).item()
                    source_ac[index]['c2'] += torch.sum(torch.round(pred2) == labels).item()
                    source_ac[index]['count'] += len(labels)

                    loss1 = self.loss_extractor(pred1, labels)
                    loss2 = self.loss_extractor(pred2, labels)

                    loss_cls += loss1 + loss2
                                
                # Moment Distance Loss
                # Second Term of equation (2)
                m1_loss = 0
                m2_loss = 0

                for k in range(1,3):
                    for i_index, batch in enumerate(source_batches):
                        
                        unlabelled_target_features = unlabelled_target_batch[0][0].to(self.device)
                        
                        features = batch[0][0]
                        labels = batch[1][0]
                        
                        features = features.to(self.device)
                        labels = labels.to(self.device)

                        src_feature = self.model(features, output_only_features = True)
                        tar_feature = self.model(unlabelled_target_features, output_only_features = True)

                        
                        e_src = torch.mean(src_feature**k, dim=0)
                        e_tar = torch.mean(tar_feature**k, dim=0)
                        m1_dist = e_src.dist(e_tar)
                        m1_loss += m1_dist

                        for j_index, other_batch in enumerate(source_batches[i_index+1:]):
                            other_x = other_batch[0][0]
                            other_y = other_batch[1][0]

                            other_x = other_x.to(self.device)
                            other_y = other_y.to(self.device)

                            other_feature = self.model(other_x, output_only_features = True)

                            e_other = torch.mean(other_feature**k, dim=0)
                            m2_dist = e_src.dist(e_other)
                            m2_loss += m2_dist

                loss_m =  (self.epochs-epoch)/self.epochs * (m1_loss/N + m2_loss/N/(N-1)*2) * 0.8

                loss = loss_cls + loss_m
                
                self.extractor_optimizer.zero_grad(set_to_none = True)

                for i in range(N):
                    self.predictor_optimizers[i].zero_grad(set_to_none = True)
                
                loss.backward()

                self.extractor_optimizer.step()
                
                for i in range(N):
                    self.predictor_optimizers[i].step()
                    self.predictor_optimizers[i].zero_grad()
                    
                self.extractor_optimizer.zero_grad()

                ######### Step ii.

                unlabelled_target_features = unlabelled_target_batch[0][0].to(self.device)
                    
                loss = 0
                d_loss = 0
                c_loss = 0

                for index, batch in enumerate(source_batches):
                    features = batch[0][0]
                    labels = batch[1][0]
                    
                    features = features.to(self.device)
                    labels = labels.to(self.device)

                    src_pred1, src_pred2 = self.model(features, index=index)
                    # First term of equation (3)
                    c_loss += self.loss_extractor(src_pred1, labels) + self.loss_extractor(src_pred2, labels)


                    tgt_pred1, tgt_pred2 = self.model(unlabelled_target_features, index=index)
                    
                    combine1 = (torch.sigmoid(tgt_pred1) + torch.sigmoid(tgt_pred2))/2
                    # Second term of equation (3)
                    d_loss += self.loss_l1(tgt_pred1, tgt_pred2)

                    for index_2, o_batch in enumerate(source_batches[index+1:]):
                        
                        pred_2_c1, pred_2_c2 = self.model(unlabelled_target_features, index = index_2+index)
                        
                        combine2 = (torch.sigmoid(pred_2_c1) + torch.sigmoid(pred_2_c2))/2

                        d_loss += self.loss_l1(combine1, combine2) * 0.1
                # Equation (3)
                loss = c_loss - d_loss 
                
                loss.backward()
                self.extractor_optimizer.zero_grad(set_to_none = True)

                for i in range(N):
                    self.predictor_optimizers[i].zero_grad()
                
                for i in range(N):
                    self.predictor_optimizers[i].step()
                
                for i in range(N):
                    self.predictor_optimizers[i].zero_grad()
                
                ########## Step iii.

                all_dis = 0

                for i in range(3):
                    discrepency_loss = 0
                    tar_feature = self.model(unlabelled_target_features, output_only_features = True)

                    for index, _ in enumerate(source_batches):
                        
                        pred_c1, pred_c2 = self.model(tar_feature, index = index, feature_extractor_input = True)
                        
                        combine1 = (torch.sigmoid(pred_c1) + torch.sigmoid(pred_c2))/2
                        # Equation (4)
                        discrepency_loss += self.loss_l1(pred_c1, pred_c2)

                        for index2, _ in enumerate(source_batches[index+1:]):
                            
                            pred_2_c1, pred_2_c2 = self.model(tar_feature, index = index2+index, feature_extractor_input = True)

                            combine2 = (torch.sigmoid(pred_2_c1) + torch.sigmoid(pred_2_c2))/2

                            discrepency_loss += self.loss_l1(combine1, combine2) * 0.1

                    all_dis += discrepency_loss.item()

                    self.extractor_optimizer.zero_grad(set_to_none = True)

                    for i in range(N):
                        self.predictor_optimizers[i].zero_grad(set_to_none = True)

                    discrepency_loss.backward()

                    self.extractor_optimizer.step()
                    self.extractor_optimizer.zero_grad(set_to_none = True)
                    
                    for i in range(N):
                        self.predictor_optimizers[i].zero_grad(set_to_none = True)
                        
        # set weights to highest source accuracy
        for i in range(N):
            c1_acc = source_ac[i]['c1']/source_ac[i]['count']
            c2_acc = source_ac[i]['c2']/source_ac[i]['count']
            self.model.weights[i] = max(c1_acc, c2_acc)

    def create_optimizer(self) -> None:
        self.extractor_optimizer = optim.Adadelta(self.model.feature_extractor.parameters(), lr=self.lr)
        self.predictor_optimizers = []
        for i in range(len(self.source_dataloader_list)):
            self.predictor_optimizers.append(optim.Adam(list(self.model.task_classifiers[i][0].parameters()) + list(self.model.task_classifiers[i][1].parameters()), lr = self.lr, betas = (self.beta1, self.beta2)))

    def create_criterion(self) -> None:
        self.loss_extractor = nn.BCEWithLogitsLoss()
        self.loss_l1 = nn.L1Loss()

    # overrides load_settings
    def load_exp_settings(self) -> None:
        self.exp_name = self.current_experiment.get("exp_name", "standard_name")   
        self.feature_extractor = self.current_experiment.get("feature_extractor", "BERT_cnn")
        self.task_classifier = self.current_experiment.get("task_classifier", "DANN_task_classifier")
        self.domain_classifier = self.current_experiment.get("domain_classifier", "DANN_domain_classifier")
        self.gamma = self.current_experiment.get("gamma", 10)
        self.mu = self.current_experiment.get("mu", 1e-2)
        self.lr = self.current_experiment.get("lr", 2.95e-5)
        self.beta1 = self.current_experiment.get("beta1", 0.92)
        self.beta2 = self.current_experiment.get("beta1", 0.993)
        
    def create_model(self):
        
        from src.model.feature_extractors import BERT_cnn
        #import .model.feature_extractors
        feature_extractor = BERT_cnn(self.bottleneck_dim)
        
        from src.model.task_classifiers import DANN_task_classifier
        task_classifier = DANN_task_classifier(self.bottleneck_dim, self.layer_size)

        self.model = M3SDA_model(feature_extractor, task_classifier, len(self.source_dataloader_list)).to(self.device)

    def create_dataloader(self):
        # fetch source datasets
        source_datasets = []
        self.unlabelled_size = 0
        for source_name in self.sources:
            train_features, val_features, train_labels, val_labels = self.fetch_dataset(source_name, labelled = True, target = False)
            self.unlabelled_size += len(train_features) + len(val_features)
            # discard validation features and labels
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