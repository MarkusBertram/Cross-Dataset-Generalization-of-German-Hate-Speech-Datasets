from typing import Dict, List, Union

# python
import datetime
import os
import json
from xml.etree.ElementPath import prepare_descendant
import numpy as np
from model.MME_model import MME_model     
from utils.utils import (fetch_import_module, get_tweet_timestamp,
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

from experiment_base import experiment_base
from utils.exp_utils import CustomConcatDataset

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
        # param_lr_g = []
        # for param_group in self.optimizer_g.param_groups:
        #     print(param_group)

        params = [{
            "params": self.model.feature_extractor.parameters(), "lr": self.lr
        }]
        self.optimizer_g = optim.SGD(
            params,
            momentum = self.momentum,
            weight_decay=self.weight_decay,
            nesterov = self.nesterov
        )
        params = [{
            "params": self.model.task_classifier.parameters(), "lr": self.lr
        }]

        self.optimizer_f = optim.SGD(
            params,
            #self.model.task_classifier.parameters(),
            #lr=self.lr,
            momentum = self.momentum,
            weight_decay=self.weight_decay,
            nesterov = self.nesterov
        )

        #     param_lr_g.append(param_group["lr"])
        # param_lr_f = []
        # for param_group in self.optimizer_f.param_groups:
        #     print(param_group)
        #     param_lr_f.append(param_group["lr"])
        
        # loss_class = torch.nn.NLLLoss().to(self.device)
        # loss_domain = torch.nn.NLLLoss().to(self.device)

        for name, param in self.model.named_parameters():
            if "bert" in name:
                param.requires_grad = False

        data_iter_s = iter(self.source_dataloader)
        data_iter_t = iter(self.labelled_target_dataloader)
        data_iter_t_unl = iter(self.unlabelled_target_dataloader)
        len_train_source = len(self.source_dataloader)
        len_train_target = len(self.labelled_target_dataloader)
        len_train_target_semi = len(self.unlabelled_target_dataloader)
        best_acc = 0
        counter = 0

        for step in range(self.steps):
            optimizer_g = inv_lr_scheduler(optimizer_g, step,
                                       init_lr=self.lr)
            optimizer_f = inv_lr_scheduler(optimizer_f, step,
                                        init_lr=self.lr)
            pass
        # for epoch in range(1, self.epochs + 1):
        #     self.model.train()
        #     total_loss = 0

        #     len_dataloader = len(self.train_dataloader)

        #     for i, (source_batch, unlabelled_target_features) in enumerate(self.train_dataloader):

        #         self.optimizer.zero_grad(set_to_none=True)
        #         p = float(i + epoch * len_dataloader) / self.epochs / len_dataloader
        #         alpha = 2. / (1. + np.exp(-10 * p)) - 1

        #         # training model using source data
        #         source_features = source_batch[0][0].to(self.device)
        #         source_labels = source_batch[1][0].to(self.device)
                
        #         #self.model.zero_grad()
        #         batch_size = len(source_labels)

        #         class_label = torch.LongTensor(batch_size).to(self.device)
        #         domain_label = torch.zeros(batch_size)
        #         domain_label = domain_label.long().to(self.device)

        #         class_label.resize_as_(source_labels).copy_(source_labels)
        #         if epoch == 1 and i == 0:
        #             self.writer.add_graph(self.model, input_to_model=[source_features, torch.tensor(alpha)], verbose=False)
        #         class_output, domain_output = self.model(input_data=source_features, alpha=alpha)
                
        #         loss_s_label = loss_class(class_output, class_label)
        #         loss_s_domain = loss_domain(domain_output, domain_label)

        #         # training model using target data
        #         unlabelled_target_features = unlabelled_target_features[0].to(self.device)

        #         batch_size = len(unlabelled_target_features)

        #         domain_label = torch.ones(batch_size)
        #         domain_label = domain_label.long().to(self.device)

        #         _, domain_output = self.model(input_data=unlabelled_target_features, alpha=alpha)
        #         loss_t_domain = loss_domain(domain_output, domain_label)
        #         loss = loss_t_domain + loss_s_domain + loss_s_label

        #         total_loss += loss.item()
                

        #         loss.backward()
        #         self.optimizer.step()

        #     self.writer.add_scalar(f"total_loss/train/{self.exp_name}", total_loss, epoch)
        #     # test after each epoch
        #     if self.test_after_each_epoch == True:
        #         self.test(epoch)

        # # add hparams
        # self.writer.add_hparams(
        #     {
        #         "lr": self.lr,

        #     },

        #     {
        #         "hparam/total_loss/train": total_loss
        #     },
        #     run_name = self.exp_name
        # )

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
        pass

    def create_criterion(self) -> None:
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    # ovlossides load_settings
    def load_exp_settings(self) -> None:
        self.exp_name = self.current_experiment.get("exp_name", "standard_name")   
        self.feature_extractor = self.current_experiment.get("feature_extractor", "BERT_cls")
        self.task_classifier = self.current_experiment.get("task_classifier", "tc1")
        self.steps = self.current_experiment.get("steps", 50000)
        self.multiplication = self.current_experiment.get("multiplication", 50000)

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

        self.model = MME_model(feature_extractor, task_classifier, output_hidden_states).to(self.device)

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
        labelled_target_train_size = int((1-self.train_split) * len(labelled_target_dataset_features))
        train_indices = random_indices[:labelled_target_train_size]
        test_indices = random_indices[labelled_target_train_size:]
        
        labelled_target_features_train = labelled_target_dataset_features[train_indices]
        labelled_target_features_test = labelled_target_dataset_features[test_indices]

        labelled_target_labels_train = labelled_target_dataset_labels[train_indices]
        labelled_target_labels_test = labelled_target_dataset_labels[test_indices]
        

        # source_features.append(labelled_target_features_train)
        # source_labels.append(labelled_target_labels_train)

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
        labelled_target_dataset = TensorDataset(labelled_target_features_train, labelled_target_labels_train)

        sampler = BatchSampler(RandomSampler(source_dataset), batch_size=min(self.batch_size, len(labelled_target_dataset)), drop_last=True)
        self.labelled_target_dataloader = DataLoader(dataset=labelled_target_dataset, sampler = sampler, num_workers=self.num_workers)            
        del labelled_target_dataset_features
        del labelled_target_dataset_labels
        gc.collect()

        # fetch unlabelled target dataset and create dataloader
        unlabelled_target_dataset_features, _ = self.fetch_dataset(self.target_unlabelled, labelled = False, target = True)
        unlabelled_target_dataset = TensorDataset(unlabelled_target_dataset_features)
        sampler = BatchSampler(RandomSampler(unlabelled_target_dataset), batch_size=2 * self.batch_size, drop_last=True)
        self.unlabelled_target_dataloader = DataLoader(dataset=unlabelled_target_dataset, sampler = sampler, num_workers=self.num_workers)            
        
        del unlabelled_target_dataset_features

        # create test dataloader
        labelled_target_dataset_test = TensorDataset(labelled_target_features_test, labelled_target_labels_test)
        sampler = BatchSampler(RandomSampler(labelled_target_dataset_test), batch_size=2*self.batch_size, drop_last=True)
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

        # create optimizer
        self.create_optimizer()

        # create criterion
        self.create_criterion()

        # perform train
        self.train()
        
        # perform test
        if self.test_after_each_epoch == False:
            self.test()

        # plot
        # self.plot()