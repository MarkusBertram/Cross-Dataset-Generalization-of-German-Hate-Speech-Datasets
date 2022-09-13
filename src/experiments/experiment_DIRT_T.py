from typing import Dict, List, Union

# python
import datetime
import os
import json
from torch import autograd
from xml.etree.ElementPath import prepare_descendant
import numpy as np
from src.model.DIRT_T_model import DIRT_T_model   
from src.utils.utils import (fetch_import_module, get_tweet_timestamp,
                         preprocess_text, print_data_example,
                         separate_text_by_classes)

import pandas as pd
from torchmetrics import F1Score
import yaml
import copy
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
from src.vat_loss import *
from src.experiments.experiment_base import experiment_base
from src.utils.exp_utils import CustomConcatDataset
import torch.nn.functional as F
from transformers import BertModel

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class WeightEMA (object):
    def __init__(self, params, src_params, alpha=0.998):

        self.params = list(params)
        self.src_params = list(src_params)
        self.alpha = alpha

        for p, src_p in zip(self.params, self.src_params):
            p.data[:] = src_p.data[:]

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.params, self.src_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)

    def zero_grad(self):
        pass 

class experiment_DIRT_T(experiment_base):
    def __init__(
        self,
        basic_settings: Dict,
        exp_settings: Dict,
        #log_path: str,
        writer: SummaryWriter,

    ):
        super(experiment_DIRT_T, self).__init__(basic_settings, exp_settings, writer)#, log_path, writer)
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
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if "bert" in name:
                    param.requires_grad = False


        ############# VADA Training

        for epoch in range(1, self.epochs + 1):
            self.model.train()

            for i, (source_batch, unlabelled_target_features) in enumerate(self.train_dataloader):
                               
                #source_features = source_batch[0][0].to(self.device)
                source_labels = source_batch[1][0].to(self.device)

                source_input_ids = source_batch[0][0][:,0].to(self.device)
                source_attention_mask = source_batch[0][0][:,1].to(self.device)

                unlabelled_target_features = unlabelled_target_features[0].to(self.device)

                source_bert_output = self.bert(input_ids=source_input_ids, attention_mask=source_attention_mask, return_dict = False, output_hidden_states=self.output_hidden_states)
                target_bert_output = self.bert(input_ids=unlabelled_target_features[:,0], attention_mask=unlabelled_target_features[:,1], return_dict = False, output_hidden_states=self.output_hidden_states)
                
                source_class_output, source_domain_output = self.model(input_features=source_bert_output)
                target_class_output, target_domain_output = self.model(input_features=target_bert_output)

                # Cross-Entropy Loss of Source Domain, Source Generalization Error
                crossE_loss = self.crossE(source_class_output, source_labels)

                # conditional entropy with respect to target distribution, enforces cluster assumption
                conditionE_loss = self.conditionE(target_class_output)               

                # Domain Discriminator, Divergence of Source and Target Domain
                domain_loss = .5*(
                self.disc(source_domain_output,torch.zeros_like(source_domain_output)) + 
                self.disc(target_domain_output, torch.ones_like(target_domain_output))
                )
                
                vat_src_loss = self.src_vat(source_bert_output, source_class_output, self.model)
                vat_tgt_loss = self.tgt_vat(target_bert_output, target_class_output, self.model)

                disc_loss = 0.5 *(
                self.disc(source_domain_output,torch.ones_like(source_domain_output)) + 
                self.disc(target_domain_output, torch.zeros_like(target_domain_output))
                )

                loss = crossE_loss +self.lambda_d*domain_loss + self.lambda_d*disc_loss +  self.lambda_s*vat_src_loss + self.lambda_t*vat_tgt_loss + self.lambda_t*conditionE_loss
                self.optimizer.zero_grad(set_to_none=True)

                loss.backward()

                self.optimizer.step()

        ############ DIRT_T Training

        self.teacher = copy.deepcopy(self.model)
        

        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.crossE      = nn.BCEWithLogitsLoss().to(self.device)
        self.conditionE  = ConditionalEntropy().to(self.device)
        self.tgt_vat     = VATLoss().to(self.device)
        self.dirt        = KLDivWithLogits()

  #     self.teacher.eval()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if "bert" in name:
                    param.requires_grad = False
        with torch.no_grad():
            for name, param in self.teacher.named_parameters():
                if "bert" in name:
                    param.requires_grad = False

        for epoch in range(1, self.epochs + 1):
            self.model.train()

            for i, (source_batch, unlabelled_target_features) in enumerate(self.train_dataloader):

                self.optimizer.zero_grad(set_to_none=True)

                unlabelled_target_features = unlabelled_target_features[0].to(self.device)
                target_bert_output = self.bert(input_ids=unlabelled_target_features[:,0], attention_mask=unlabelled_target_features[:,1], return_dict = False, output_hidden_states=self.output_hidden_states)

                if self.output_hidden_states == False:
                    target_bert_output = target_bert_output[0][:,0,:]

                target_class_output, target_domain_output = self.model(input_features=target_bert_output)
                teacher_target_class_output, teacher_target_domain_output = self.teacher(input_features=target_bert_output)

                conditionE_loss = self.conditionE(target_class_output) # conditional entropy
                dirt_loss       = self.dirt(target_class_output, teacher_target_class_output)

                vat_tgt_loss    = self.tgt_vat(target_bert_output, target_class_output, self.model)

                loss = self.lambda_t*conditionE_loss + self.lambda_t*vat_tgt_loss + self.beta*dirt_loss 
                
                loss.backward()
                self.optimizer.step()

            # polyak averaging
            # https://discuss.pytorch.org/t/copying-weights-from-one-net-to-another/1492/17
            for target_param, param in zip(self.teacher.parameters(), self.model.parameters()):
                target_param.data.copy_(self.polyak_factor*param.data + target_param.data*(1.0 - self.polyak_factor))

    def create_optimizer(self) -> None:
        params = list(self.model.feature_extractor.parameters()) + list(self.model.task_classifier.parameters()) + list(self.model.domain_classifier.parameters()) 
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(self.beta1, self.beta2))
           
    def create_criterion(self) -> None:
        self.crossE      = nn.BCEWithLogitsLoss().to(self.device)
        self.conditionE  = ConditionalEntropy().to(self.device)
        self.src_vat     = VATLoss().to(self.device)
        self.tgt_vat     = VATLoss().to(self.device)
        self.disc        = nn.BCEWithLogitsLoss().to(self.device)

    # overrides load_settings
    def load_exp_settings(self) -> None:
        self.exp_name = self.current_experiment.get("exp_name", "DIRT_T_cnn")
        self.lr = self.current_experiment.get("lr", 1.4e-05)
        self.beta = self.current_experiment.get("beta", 0.0006)
        self.beta1 = self.current_experiment.get("beta1", 0.913)
        self.beta2 = self.current_experiment.get("beta1", 0.993)
        self.radius = self.current_experiment.get("radius", 1)
        self.lambda_d = self.current_experiment.get("lambda_d", 0.006)
        self.lambda_s = self.current_experiment.get("lambda_s", 0.774)
        self.lambda_t = self.current_experiment.get("lambda_t", 0.016)
        self.polyak_factor = self.current_experiment.get("polyak_factor", 0.998)

    def create_model(self):
        
        from src.model.feature_extractors import BERT_cnn
        #import .model.feature_extractors
        feature_extractor = BERT_cnn(self.bottleneck_dim)
        self.output_hidden_states = True

        from src.model.task_classifiers import DANN_task_classifier
        task_classifier = DANN_task_classifier(self.bottleneck_dim, self.layer_size)
            
        from src.model.domain_classifiers import DANN_domain_classifier
        domain_classifier = DANN_domain_classifier(self.bottleneck_dim, self.layer_size)
        
        self.bert = BertModel.from_pretrained("deepset/gbert-base").to(self.device)
        self.model = DIRT_T_model(feature_extractor, task_classifier, domain_classifier, self.output_hidden_states).to(self.device)

    def create_dataloader(self):
        # fetch source datasets
        source_features = []
        source_labels = []
        self.unlabelled_size = 0
        for source_name in self.sources:
            train_features, val_features, train_labels, val_labels = self.fetch_dataset(source_name, labelled = True, target = False)
            self.unlabelled_size += len(train_features) + len(val_features)
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

        # create criterion
        self.create_criterion()

        # create optimizer
        self.create_optimizer()       

        # perform train
        self.train()
        
        # perform test
        self.test()