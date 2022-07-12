from typing import Dict, List, Union

# python
import datetime
import os
import json
from xml.etree.ElementPath import prepare_descendant
import numpy as np
from model.DANN_model import DANN_model     
from utils.utils import (fetch_import_module, get_tweet_timestamp,
                         preprocess_text, print_data_example,
                         separate_text_by_classes)

import pandas as pd
from pandarallel import pandarallel
from torchmetrics import F1Score
import yaml
from torch.utils.tensorboard.writer import SummaryWriter
from sklearn.model_selection import train_test_split
import sys
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

class CustomConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

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

        # lr_sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     "min",
        #     factor=0.1,
        #     patience=int(self.epochs * 0.05),
        #     min_lr=1e-7,
        # )

        loss_class = torch.nn.NLLLoss().to(self.device)
        loss_domain = torch.nn.NLLLoss().to(self.device)

        for name, param in self.model.named_parameters():
            if "bert" in name:
                param.requires_grad = False

        self.model.train()

        for epoch in range(1, self.epochs + 1):

            total_loss = 0

            len_dataloader = len(self.train_dataloader)

            for i, (source_batch, unlabelled_target_features) in enumerate(self.train_dataloader):

                self.optimizer.zero_grad(set_to_none=True)
                p = float(i + epoch * len_dataloader) / self.epochs / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                # training model using source data
                source_features = source_batch[0][0].to(self.device)
                source_labels = source_batch[1][0].to(self.device)
                
                self.model.zero_grad()
                batch_size = len(source_labels)

                class_label = torch.LongTensor(batch_size).to(self.device)
                domain_label = torch.zeros(batch_size)
                domain_label = domain_label.long().to(self.device)

                class_label.resize_as_(source_labels).copy_(source_labels)
                if epoch == 0 and i == 0:
                    self.writer.add_graph(self.model, input_to_model=[source_features, alpha], verbose=False)
                class_output, domain_output = self.model(input_data=source_features, alpha=alpha)
                
                loss_s_label = loss_class(class_output, class_label)
                loss_s_domain = loss_domain(domain_output, domain_label)

                # training model using target data
                unlabelled_target_features = unlabelled_target_features[0].to(self.device)

                batch_size = len(unlabelled_target_features)

                domain_label = torch.ones(batch_size)
                domain_label = domain_label.long().to(self.device)

                _, domain_output = self.model(input_data=unlabelled_target_features, alpha=alpha)
                loss_t_domain = loss_domain(domain_output, domain_label)
                loss = loss_t_domain + loss_s_domain + loss_s_label

                total_loss += loss.item()
                

                loss.backward()
                self.optimizer.step()

                i += 1

            self.writer.add_scalar("total_loss/train", total_loss, epoch)

        # add hparams
        self.writer.add_hparams(
            {
                "lr": self.lr,

            },

            {
                "hparam/total_loss/train": total_loss
            },
            run_name = self.exp_name
        )

        self.test(epoch)

    # ovlossides test
    @torch.no_grad()
    def test(self, epoch):
        """test [computes loss of the test set]
        [extended_summary]
        Returns:
            [type]: [description]
        """

        correct = 0
        predictions = []
        targets = []
        f1 = F1Score(num_class = 2)
        self.model.eval()
        for (target_features, target_labels) in self.test_dataloader:
            target_features = target_features.to(self.device)
            target_labels = target_labels.to(self.device)
            
            target_class_output, _ = self.model(target_features)
            
            target_class_predictions = torch.argmax(target_class_output, dim=1)

            predictions.append(target_class_predictions)
            targets.append(target_labels)
            #f1_score = f1(preds, target_labels)

            correct += torch.sum(target_class_predictions == target_labels).item()

        avg_test_acc = correct / len(self.test_dataloader.dataset)

        outputs = np.concatenate(predictions)
        targets = np.concatenate(targets)
        f1score = f1(outputs, targets, average = "macro")
        
        self.writer.add_scalar("Accuracy/Test", avg_test_acc, epoch)
        self.writer.add_scalar("F1_score/Test", f1score, epoch)

    def create_optimizer(self) -> None:
        self.optimizer = optim.Adam(
            self.model.parameters(),
            weight_decay=self.weight_decay,
            lr=self.lr,
        )

    def create_criterion(self) -> None:
        self.criterion = nn.CrossEntropyLoss()

    # ovlossides load_settings
    def load_exp_settings(self) -> None:
        self.exp_name = self.current_experiment.get("exp_name", "standard_name")   
        self.feature_extractor = self.current_experiment.get("feature_extractor", "BERT_cls")
        self.task_classifier = self.current_experiment.get("task_classifier", "tc1")
        self.domain_classifier = self.current_experiment.get("domain_classifier", "dc1")
        
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
            print("lossor, can't find this feature extractor. please specify bert_cls or bert_cnn in experiment settings.")
        
        if self.task_classifier.lower() == "tc1":
            from model.task_classifiers import task_classifier1
            task_classifier = task_classifier1()
            
        if self.domain_classifier.lower() == "dc1":
            from model.domain_classifiers import domain_classifier1
            domain_classifier = domain_classifier1()

        self.model = DANN_model(feature_extractor, task_classifier, domain_classifier, output_hidden_states).to(self.device)

    def create_dataloader(self):
        pandarallel.initialize(nb_workers = self.num_workers, progress_bar=True)
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

        # create optimizer
        self.create_optimizer()

        # create criterion
        self.create_criterion()

        # perform train
        self.train()
        
        # perform test
        #self.test()

        # plot
        # self.plot()