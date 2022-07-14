from typing import Dict, List, Union

# python
import datetime
import os
import json
from xml.etree.ElementPath import prepare_descendant
import numpy as np
from model.labelled_only_model import labelled_only_model
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

class experiment_target_only(experiment_base):
    def __init__(
        self,
        basic_settings: Dict,
        exp_settings: Dict,
        #log_path: str,
        writer: SummaryWriter,

    ):
        super(experiment_target_only, self).__init__(basic_settings, exp_settings, writer)#, log_path, writer)
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

        loss_class = torch.nn.NLLLoss().to(self.device)

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

                if epoch == 1 and i == 0:
                    self.writer.add_graph(self.model, input_to_model=[target_features], verbose=False)
                
                class_output = self.model(input_data=target_features)
                
                loss_s_label = loss_class(class_output, target_labels)

                loss = loss_s_label

                total_loss += loss.item()
                
                loss.backward()
                self.optimizer.step()

            self.writer.add_scalar(f"total_loss/train/{self.exp_name}", total_loss, epoch)
            # test after each epoch
            if self.test_after_each_epoch == True:
                self.test(epoch)

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

    # ovlossides test
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

            target_class_output = self.model(target_features)
    
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

        self.model = labelled_only_model(feature_extractor, task_classifier, output_hidden_states).to(self.device)

    def create_dataloader(self):
                
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
        
        labelled_target_dataset_train = TensorDataset(labelled_target_features_train, labelled_target_labels_train)

        del labelled_target_dataset_features
        del labelled_target_dataset_labels
        gc.collect()
        # create train dataloader
        sampler = BatchSampler(RandomSampler(labelled_target_dataset_train), batch_size=self.batch_size, drop_last=False)
        self.train_dataloader = DataLoader(dataset=labelled_target_dataset_train, sampler = sampler, num_workers=self.num_workers)            

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
        if self.test_after_each_epoch == False:
            self.test()

        # plot
        # self.plot()