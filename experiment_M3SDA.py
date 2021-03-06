from typing import Dict, List, Union

# python
import datetime
import os
import json
from xml.etree.ElementPath import prepare_descendant
import numpy as np
from responses import target
from model.M3SDA_model import M3SDA_model     
from utils.utils import (fetch_import_module, get_tweet_timestamp,
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
from experiment_base import experiment_base
from utils.exp_utils import CustomConcatDataset

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

        source_loss = {}
        for i in range(N):
            source_loss[i] = {}
            for j in range(1, 3):
                source_loss[i][str(j)] = {}
                source_loss[i][str(j)]['loss'] = []
                source_loss[i][str(j)]['ac'] = []
        
        min_ = len(self.unlabelled_target_dataloader)

        for epoch in range(1, self.epochs + 1):
            self.model.train()

            source_ac = {}
            for i in range(N):
                source_ac[i] = defaultdict(int)

            record = {}
            for i in range(N):
                record[i] = {}
                for j in range(1, 3):
                    record[i][str(j)] = 0
            mcd_loss = 0
            dis_loss = 0

            train_dataloader = multi_data_loader(self.source_dataloader_list, self.unlabelled_target_dataloader)

            for batch_index, (source_batches, unlabelled_target_batch) in enumerate(train_dataloader):

                loss_cls = 0
                src_len = len(source_batches)
                
                # train extractor and source classifier
                for index, batch in enumerate(source_batches):
                    features = batch[0][0]
                    labels = batch[1][0]
                    
                    features = features.to(self.device)
                    labels = labels.to(self.device)

                    pred1, pred2 = self.model(features, index)

                    source_ac[index]['c1'] += torch.sum(torch.max(pred1, dim=1)[1] == labels).item()
                    source_ac[index]['c2'] += torch.sum(torch.max(pred2, dim=1)[1] == labels).item()
                    source_ac[index]['count'] += len(labels)

                    loss1 = self.loss_extractor(pred1, labels)
                    loss2 = self.loss_extractor(pred2, labels)

                    loss_cls += loss1 + loss2
                
                if self.verbose:
                    if batch_index % 10 == 0:
                        for i in range(N):
                                print('c1 : [%.8f]' % (source_ac[i]['c1']/(batch_index+1)/self.batch_size))
                                print('c2 : [%.8f]' % (source_ac[i]['c2']/(batch_index+1)/self.batch_size))
                                
                
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
                mcd_loss += loss_m.item()

                loss = loss_cls + loss_m

                if batch_index % 10 == 0:
                    print('[%d]/[%d]' % (batch_index, min_))
                    print('class loss : [%.5f]' % (loss_cls))
                    print('msd loss : [%.5f]' % (loss_m))
                
                self.extractor_optimizer.zero_grad(set_to_none = True)

                for i in range(N):
                    self.predictor_optimizers[i].zero_grad(set_to_none = True)
                
                loss.backward()

                self.extractor_optimizer.step()
                
                for i in range(N):
                    self.predictor_optimizers[i].step()
                    self.predictor_optimizers[i].zero_grad()
                    
                self.extractor_optimizer.zero_grad()

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

                    c_loss += self.loss_extractor(src_pred1, labels) + self.loss_extractor(src_pred2, labels)


                    tgt_pred1, tgt_pred2 = self.model(unlabelled_target_features, index=index)

                    combine1 = (F.softmax(tgt_pred1, dim=1) + F.softmax(tgt_pred2, dim=1))/2

                    d_loss += self.loss_l1(tgt_pred1, tgt_pred2)

                    for index_2, o_batch in enumerate(source_batches[index+1:]):
                        
                        pred_2_c1, pred_2_c2 = self.model(unlabelled_target_features, index = index_2+index)
                        
                        combine2 = (F.softmax(pred_2_c1, dim=1) + F.softmax(pred_2_c2, dim=1))/2

                        d_loss += self.loss_l1(combine1, combine2) * 0.1

                loss = c_loss - d_loss 
                
                loss.backward()
                self.extractor_optimizer.zero_grad(set_to_none = True)

                for i in range(N):
                    self.predictor_optimizers[i].zero_grad()
                
                for i in range(N):
                    self.predictor_optimizers[i].step()
                
                for i in range(N):
                    self.predictor_optimizers[i].zero_grad()
                
                all_dis = 0

                for i in range(3):
                    discrepency_loss = 0
                    tar_feature = self.model(unlabelled_target_features, output_only_features = True)

                    for index, _ in enumerate(source_batches):
                        
                        pred_c1, pred_c2 = self.model(tar_feature, index = index, feature_extractor_input = True)
                        
                        combine1 = (F.softmax(pred_c1, dim=1) + F.softmax(pred_c2, dim=1))/2

                        discrepency_loss += self.loss_l1(pred_c1, pred_c2)

                        for index2, _ in enumerate(source_batches[index+1:]):
                            
                            pred_2_c1, pred_2_c2 = self.model(tar_feature, index = index2+index, feature_extractor_input = True)

                            combine2 = (F.softmax(pred_2_c1, dim=1) + F.softmax(pred_2_c2, dim=1))/2

                            discrepency_loss += self.loss_l1(combine1, combine2) * 0.1
                        #discrepency_loss += torch.mean(torch.sum(abs(F.softmax(pred_c1, dim=1) - F.softmax(pred_c2, dim=1)), dim=1))
                        #discrepency_loss += loss_l1(F.softmax(pred_c1, dim=1), F.softmax(pred_c2, dim=1)) 

                    all_dis += discrepency_loss.item()

                    self.extractor_optimizer.zero_grad(set_to_none = True)

                    for i in range(N):
                        self.predictor_optimizers[i].zero_grad(set_to_none = True)

                    discrepency_loss.backward()

                    self.extractor_optimizer.step()
                    self.extractor_optimizer.zero_grad(set_to_none = True)
                    
                    for i in range(N):
                        self.predictor_optimizers[i].zero_grad(set_to_none = True)

                dis_loss += all_dis

                if self.verbose:
                    if batch_index % 10 == 0:
                        print('Discrepency Loss : [%.4f]' % (all_dis))

        # set weights to highest source accuracy
        for i in range(N):
            c1_acc = source_ac[i]['c1']/source_ac[i]['count']
            c2_acc = source_ac[i]['c2']/source_ac[i]['count']
            self.model.weights[i] = max(c1_acc, c2_acc)

    def create_optimizer(self) -> None:
        self.extractor_optimizer = optim.Adadelta(self.model.feature_extractor.parameters(), lr=self.lr)
        self.predictor_optimizers = []
        for i in range(len(self.source_dataloader_list)):
            self.predictor_optimizers.append(optim.Adam(list(self.model.task_classifiers[i][0].parameters()) + list(self.model.task_classifiers[i][1].parameters()), lr = self.lr))

    def create_criterion(self) -> None:
        self.loss_extractor = nn.CrossEntropyLoss()
        self.loss_l1 = nn.L1Loss()
        self.loss_l2 = nn.MSELoss()

    # overrides load_settings
    def load_exp_settings(self) -> None:
        self.exp_name = self.current_experiment.get("exp_name", "standard_name")   
        self.feature_extractor = self.current_experiment.get("feature_extractor", "BERT_cls")
        self.task_classifier = self.current_experiment.get("task_classifier", "tc1")
        self.domain_classifier = self.current_experiment.get("domain_classifier", "dc1")
        self.gamma = self.current_experiment.get("gamma", 10)
        self.mu = self.current_experiment.get("mu", 1e-2)

    def create_model(self):
        
        if self.feature_extractor.lower() == "bert_cls":
            from model.feature_extractors import BERT_cls
            #import .model.feature_extractors
            feature_extractor = BERT_cls()
            output_hidden_states = False
        elif self.feature_extractor.lower() == "bert_cnn":
            from model.feature_extractors import BERT_cnn
            feature_extractor = BERT_cnn()
            output_hidden_states = True
        else:
            raise ValueError("Can't find the feature extractor name. \
            Please specify bert_cls or bert_cnn as key in experiment settings of the current experiment.")
        

        if self.task_classifier.lower() == "tc1":
            from model.task_classifiers import task_classifier1
            task_classifier = task_classifier1()
        else:
            raise ValueError("Can't find the task classifier name. \
            Please specify the task classifier class name as key in experiment settings of the current experiment.")
            
        if self.domain_classifier.lower() == "dc1":
            from model.domain_classifiers import domain_classifier1
            domain_classifier = domain_classifier1()
        else:
            raise ValueError("Can't find the domain classifier name. \
            Please specify the domain classifier class name as key in experiment settings of the current experiment.")

        self.model = M3SDA_model(feature_extractor, task_classifier, output_hidden_states, len(self.source_dataloader_list)).to(self.device)

    def create_dataloader(self):
        # fetch source datasets
        source_datasets = []
        for source_name in self.sources:
            features, labels = self.fetch_dataset(source_name, labelled = True, target = False)
            source_datasets.append(TensorDataset(features, labels))
        
        labelled_target_features_train, labelled_target_labels_train, labelled_target_features_test, labelled_target_labels_test = self.get_target_dataset()

        # add labelled target train dataset to source domains
        source_datasets.append(TensorDataset(labelled_target_features_train, labelled_target_labels_train))
        
        gc.collect()

        # create source dataloaders
        self.source_dataloader_list = []
        for source_dataset in source_datasets:
            sampler = BatchSampler(RandomSampler(source_dataset), batch_size=self.batch_size, drop_last=True)
            dataloader = DataLoader(dataset=source_dataset, sampler = sampler, num_workers=self.num_workers)            
            self.source_dataloader_list.append(dataloader)

        # create unlabelled target dataloader
        unlabelled_target_dataset_features, _ = self.fetch_dataset(self.target_unlabelled, labelled = False, target = True)

        # fetch unlabelled target dataset and create dataloader
        unlabelled_target_dataset_features, _ = self.fetch_dataset(self.target_unlabelled, labelled = False, target = True)
        unlabelled_target_dataset = TensorDataset(unlabelled_target_dataset_features)
        sampler = BatchSampler(RandomSampler(unlabelled_target_dataset), batch_size=self.batch_size, drop_last=True)
        self.unlabelled_target_dataloader = DataLoader(dataset=unlabelled_target_dataset, sampler = sampler, num_workers=self.num_workers)            
        
        del unlabelled_target_dataset_features

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