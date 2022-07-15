from typing import Dict, List, Union

# python
import datetime
import os
import json
from torch import autograd
from xml.etree.ElementPath import prepare_descendant
import numpy as np
from model.DIRT_T_model import DIRT_T_model   
from utils.utils import (fetch_import_module, get_tweet_timestamp,
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
from vat_loss import *
from experiment_base import experiment_base
from utils.exp_utils import CustomConcatDataset
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
        
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True       

    # ovlossides train
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


        ############# VADA Training

        for epoch in range(1, self.epochs + 1):
            self.model.train()

            for i, (source_batch, unlabelled_target_features) in enumerate(self.train_dataloader):
               
                noise = True
                
                #source_features = source_batch[0][0].to(self.device)
                source_labels = source_batch[1][0].to(self.device)

                source_input_ids = source_batch[0][0][:,0].to(self.device)
                source_attention_mask = source_batch[0][0][:,1].to(self.device)

                unlabelled_target_features = unlabelled_target_features[0].to(self.device)

                source_bert_output = self.bert(input_ids=source_input_ids, attention_mask=source_attention_mask, return_dict = False, output_hidden_states=self.output_hidden_states)
                target_bert_output = self.bert(input_ids=unlabelled_target_features[:,0], attention_mask=unlabelled_target_features[:,1], return_dict = False, output_hidden_states=self.output_hidden_states)

                if self.output_hidden_states == False:
                    source_bert_output = source_bert_output[0][:,0,:]
                    target_bert_output = target_bert_output[0][:,0,:]
                
                if epoch == 1 and i == 0:
                    self.writer.add_graph(self.model, input_to_model=[source_bert_output, torch.tensor(noise)], verbose=False)
                
                source_class_output, source_domain_output = self.model(input_features=source_bert_output, noise=noise)
                target_class_output, target_domain_output = self.model(input_features=target_bert_output, noise=noise)

                # Cross-Entropy Loss of Source Domain, Source Generalization Error
                crossE_loss     = self.crossE(source_class_output, source_labels)

                # conditional entropy with respect to target distribution, enforces cluster assumption
                conditionE_loss = self.conditionE(target_class_output)               

                # Domain Discriminator, Divergence of Source and Target Domain
                domain_loss     = .5*self.disc(source_domain_output,torch.zeros_like(source_domain_output)) + 0.5*self.disc(target_domain_output, torch.ones_like(target_domain_output))
                
                vat_src_loss    = self.src_vat(source_bert_output, source_class_output, self.model, noise)
                vat_tgt_loss    = self.tgt_vat(target_bert_output, target_class_output, self.model, noise)

                disc_loss = 0.5*self.disc(source_domain_output,torch.ones_like(source_domain_output)) + 0.5*self.disc(target_domain_output, torch.zeros_like(target_domain_output))

                loss = crossE_loss +self.lambda_d*domain_loss + self.lambda_d*disc_loss +  self.lambda_s*vat_src_loss + self.lambda_t*vat_tgt_loss + self.lambda_t*conditionE_loss
                self.optimizer.zero_grad(set_to_none=True)
                
                #disc_loss = 0.5*self.disc(source_domain_output,torch.ones_like(source_domain_output)) + 0.5*self.disc(target_domain_output, torch.zeros_like(target_domain_output))
                #self.optimizer2.zero_grad(set_to_none=True)
                
                loss.backward()
                #disc_loss.backward()

                self.optimizer.step()
                #self.optimizer2.step()


        ############ DIRT_T Training

        self.teacher = copy.deepcopy(self.model)
        
        student_params   = list(self.model.parameters())
        teacher_params   = list(self.teacher.parameters())

        for param in teacher_params:
            param.requires_grad = False

        self.optimizer   = optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizer2  = WeightEMA(teacher_params, student_params)#DelayedWeight(teacher_params, student_params)
        
        self.crossE      = nn.CrossEntropyLoss().to(self.device)
        self.conditionE  = ConditionalEntropy().to(self.device)
        self.tgt_vat     = VATLoss().to(self.device)#VATLoss(self.model, radius=self.radius).to(self.device)
        self.dirt        = KLDivWithLogits() #F.kl_div().to(self.device)#KLDivWithLogits()    

        self.model.train(True)
  #     self.teacher.eval()

        for name, param in self.model.named_parameters():
            if "bert" in name:
                param.requires_grad = False

        for name, param in self.teacher.named_parameters():
            if "bert" in name:
                param.requires_grad = False

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0

            len_dataloader = len(self.train_dataloader)

            for i, (source_batch, unlabelled_target_features) in enumerate(self.train_dataloader):

                self.optimizer.zero_grad(set_to_none=True)
                self.optimizer2.zero_grad(set_to_none=True)
                noise = True

                unlabelled_target_features = unlabelled_target_features[0].to(self.device)
                target_bert_output = self.bert(input_ids=unlabelled_target_features[:,0], attention_mask=unlabelled_target_features[:,1], return_dict = False, output_hidden_states=self.output_hidden_states).to(self.device)

                if self.output_hidden_states == False:
                    target_bert_output = target_bert_output[0][:,0,:]

                target_class_output, target_domain_output = self.model(input_features=target_bert_output, noise=noise)
                teacher_target_class_output, teacher_target_domain_output = self.teacher(input_features=target_bert_output, noise=noise)

                conditionE_loss = self.conditionE(target_class_output) # condition entropy
                dirt_loss       = self.dirt(target_class_output, teacher_target_class_output)
                #vat_tgt_loss    = self.tgt_vat(unlabelled_target_features,target_class_output)
                vat_tgt_loss    = self.tgt_vat(target_bert_output, target_class_output, self.model, noise)

                loss = self.lambda_t*conditionE_loss + self.lambda_t*vat_tgt_loss + self.beta_t*dirt_loss 
                loss.backward()
                self.optimizer.step()
                self.optimizer2.step()

    # overrides test
    @torch.no_grad()
    def test(self, epoch):
        """test [computes loss of the test set]
        [extended_summary]
        Returns:
            [type]: [description]
        """
        alpha = 0
        correct = 0
        predictions = []
        targets = []
        f1 = F1Score(num_classes = 2, average="macro")
        self.model.eval()
        for (target_features, target_labels) in self.test_dataloader:
            target_features = target_features[0].to(self.device)
            target_labels = target_labels[0].to(self.device)

            target_class_output, target_domain_output = self.model(target_features, alpha)
            
            target_class_predictions = torch.argmax(target_class_output, dim=1)

            predictions.append(target_class_predictions.cpu())
            targets.append(target_labels.cpu())
            #f1_score = f1(preds, target_labels)

            correct += torch.sum(target_class_predictions == target_labels).item()

        avg_test_acc = correct / len(self.test_dataloader.dataset)

        outputs = torch.cat(predictions)
        targets = torch.cat(targets)
        f1score = f1(outputs, targets)

        self.writer.add_scalar(f"Accuracy/Test/{self.exp_name}", avg_test_acc, epoch)
        self.writer.add_scalar(f"F1_score/Test/{self.exp_name}", f1score.item(), epoch)

        if epoch == self.epochs:
            # add hparams
            self.writer.add_hparams(
                {
                    "lr": self.lr,

                },

                {
                    "hparam/Accuracy/Test": avg_test_acc,
                    "F1_score/Test": f1score.item()
                },
                run_name = self.exp_name
            )

    def create_optimizer(self) -> None:
        params = list(self.model.feature_extractor.parameters()) + list(self.model.task_classifier.parameters()) + list(self.model.domain_classifier.parameters()) 
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(self.beta1, 0.999))

        params = list(self.model.feature_extractor.parameters()) + list(self.model.domain_classifier.parameters())
        self.optimizer2 = optim.Adam(params, lr = self.lr, betas=(self.beta1, 0.999))   
           
    def create_criterion(self) -> None:
        self.crossE      = nn.CrossEntropyLoss().to(self.device)
        self.conditionE  = ConditionalEntropy().to(self.device)
        self.src_vat     = VATLoss().to(self.device)
        self.tgt_vat     = VATLoss().to(self.device)
        self.disc        = nn.BCEWithLogitsLoss().to(self.device)

    # ovlossides load_settings
    def load_exp_settings(self) -> None:
        self.exp_name = self.current_experiment.get("exp_name", "standard_name")   
        self.feature_extractor = self.current_experiment.get("feature_extractor", "BERT_cls")
        self.task_classifier = self.current_experiment.get("task_classifier", "tc1")
        self.domain_classifier = self.current_experiment.get("domain_classifier", "dc1")
        self.beta1 = self.current_experiment.get("beta1", 0.01)
        self.radius = self.current_experiment.get("radius", 0.01)
        self.lambda_d = self.current_experiment.get("lambda_d", 0.01)
        self.lambda_s = self.current_experiment.get("lambda_s", 0.01)
        self.lambda_t = self.current_experiment.get("lambda_t", 0.01)
        self.xi = self.current_experiment.get("xi", 0.01)
        self.eps = self.current_experiment.get("eps", 0.01)
        self.ip = self.current_experiment.get("ip", 0.01)
        self.beta_t = self.current_experiment.get("beta_t", 0.01)

    def create_model(self):
        
        if self.feature_extractor.lower() == "bert_cls":
            from model.feature_extractors import BERT_cls_token_input
            #import .model.feature_extractors
            feature_extractor = BERT_cls_token_input()
            self.output_hidden_states = False
        elif self.feature_extractor.lower() == "bert_cnn":
            from model.feature_extractors import BERT_cnn
            feature_extractor = BERT_cnn()
            self.output_hidden_states = True
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
        self.bert = BertModel.from_pretrained("deepset/gbert-base").to(self.device)
        self.model = DIRT_T_model(feature_extractor, task_classifier, domain_classifier).to(self.device)

    def create_dataloader(self):
        # fetch source datasets
        source_features = []
        source_labels = []
        for source_name in self.sources:
            features, labels = self.fetch_dataset(source_name, labelled = True, target = False)
            source_features.append(features)
            source_labels.append(labels)
            
        # fetch labelled target dataset
        labelled_target_dataset_features, labelled_target_dataset_labels = self.fetch_dataset(self.target_labelled, labelled = True, target = True)

        indices = np.arange(len(labelled_target_dataset_features))
        
        random_indices = np.random.permutation(indices)
    
        # split labelled target dataset into train and test
        labelled_target_train_size = int(self.train_split * len(labelled_target_dataset_features))
        train_indices = random_indices[:labelled_target_train_size]
        test_indices = random_indices[labelled_target_train_size:]
        
        labelled_target_features_train = labelled_target_dataset_features[train_indices]
        labelled_target_features_test = labelled_target_dataset_features[test_indices]

        labelled_target_labels_train = labelled_target_dataset_labels[train_indices]
        labelled_target_labels_test = labelled_target_dataset_labels[test_indices]
        
        del labelled_target_dataset_features
        del labelled_target_dataset_labels
        gc.collect()
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
        unlabelled_target_dataset_features, _ = self.fetch_dataset(self.target_unlabelled, labelled = False, target = True)
        
        # combine source dataset and unlabelled target dataset into one dataset
        concatenated_train_dataset = CustomConcatDataset(source_dataset, unlabelled_target_dataset_features)
        sampler = BatchSampler(RandomSampler(concatenated_train_dataset), batch_size=self.batch_size, drop_last=False)
        self.train_dataloader = DataLoader(dataset=concatenated_train_dataset, sampler = sampler, num_workers=self.num_workers)            
        #self.train_dataloader = DataLoader(concatenated_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        del concatenated_train_dataset
        del unlabelled_target_dataset_features
        # create test dataloader
        
        labelled_target_dataset_test = TensorDataset(labelled_target_features_test, labelled_target_labels_test)
        sampler = BatchSampler(RandomSampler(labelled_target_dataset_test), batch_size=self.batch_size, drop_last=False)
        self.test_dataloader = DataLoader(dataset=labelled_target_dataset_test, sampler = sampler, num_workers=self.num_workers)               
        del labelled_target_dataset_test
        gc.collect()

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

        # create criterion
        self.create_criterion()

        

        # create optimizer
        self.create_optimizer()       

        # perform train
        #self.vada_train()
        self.train()
        # perform test
        if self.test_after_each_epoch == False:
            self.test()

        #self.DIRT_T_train()
        # plot
        # self.plot()