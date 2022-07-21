from typing import Dict, List, Union

# python
import datetime
import os
import json
from xml.etree.ElementPath import prepare_descendant
import numpy as np
from model.M3SDA_model import M3SDA_model     
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

        N = len(self.source_dataloader_list)

        print(N)
        sys.exit(0)

        for name, param in self.model.named_parameters():
            if "bert" in name:
                param.requires_grad = False

        source_clf_optimizers = {}
        for i in range():
            pass
        source_loss = {}

        for epoch in range(1, self.epochs + 1):
            self.model.train()

            train_dataloader = multi_data_loader(self.source_dataloader_list, self.unlabelled_target_dataloader)

            for i, (source_batches, unlabelled_target_batch) in enumerate(train_dataloader):
                self.optimizer.zero_grad()

                source_feature_list = []
                source_labels_list = []

                for i in range(len(self.source_dataloader_list)):
                    source_feature_list.append(source_batches[i][0][0].to(self.device))
                    source_labels_list.append(source_batches[i][1][0].to(self.device))
                
                unlabelled_target_features = unlabelled_target_batch[0][0].to(self.device)
                
                slabels = torch.ones(self.batch_size, requires_grad=False).type(torch.LongTensor).to(self.device)
                tlabels = torch.zeros(self.batch_size, requires_grad=False).type(torch.LongTensor).to(self.device)

                logprobs, sdomains, tdomains = self.model(source_feature_list, unlabelled_target_features)

                # Compute prediction accuracy on multiple training sources.
                losses = torch.stack([F.nll_loss(logprobs[j], source_labels_list[j]) for j in range(len(self.source_dataloader_list))])

                domain_losses = torch.stack([F.nll_loss(sdomains[j], slabels) +
                                           F.nll_loss(tdomains[j], tlabels) for j in range(len(self.source_dataloader_list))])

                loss = torch.log(torch.sum(torch.exp(self.gamma * (losses + self.mu * domain_losses)))) / self.gamma

                loss.backward()
                self.optimizer.step()

    # # ovlossides test
    # @torch.no_grad()
    # def test(self, epoch):
    #     """test [computes loss of the test set]
    #     [extended_summary]
    #     Returns:
    #         [type]: [description]
    #     """
    #     correct = 0
    #     predictions = []
    #     targets = []
    #     f1 = F1Score(num_classes = 2, average="macro")
    #     self.model.eval()
    #     for (target_features, target_labels) in self.test_dataloader:
    #         target_features = target_features[0].to(self.device)
    #         target_labels = target_labels[0].to(self.device)

    #         target_class_output = self.model.inference(target_features)
            
    #         target_class_predictions = torch.argmax(target_class_output, dim=1)

    #         predictions.append(target_class_predictions.cpu())
    #         targets.append(target_labels.cpu())
    #         #f1_score = f1(preds, target_labels)

    #         correct += torch.sum(target_class_predictions == target_labels).item()

    #     avg_test_acc = correct / len(self.test_dataloader.dataset)

    #     outputs = torch.cat(predictions)
    #     targets = torch.cat(targets)
    #     f1score = f1(outputs, targets)

    #     self.writer.add_scalar(f"Accuracy/Test/{self.exp_name}", avg_test_acc, epoch)
    #     self.writer.add_scalar(f"F1_score/Test/{self.exp_name}", f1score.item(), epoch)

    #     if epoch == self.epochs:
    #         # add hparams
    #         self.writer.add_hparams(
    #             {
    #                 "lr": self.lr,

    #             },

    #             {
    #                 "hparam/Accuracy/Test": avg_test_acc,
    #                 "F1_score/Test": f1score.item()
    #             },
    #             run_name = self.exp_name
    #         )

    def create_optimizer(self) -> None:
        self.extractor_optimizer = optim.Adadelta(self.model.feature_extractor.parameters(), lr=self.lr)
        self.predictor_optimizers = []
        for i in range(len(self.source_dataloader_list)):
            self.predictor_optimizers.append(optim.Adam(list(self.model.task_classifiers[i][0].parameters()) + list(self.model.task_classifiers[i][1].parameters()), lr = self.lr))

    def create_criterion(self) -> None:
        self.loss_extractor = nn.CrossEntropyLoss()
        self.loss_l1 = nn.L1Loss()
        self.loss_l2 = nn.MSELoss()

    # ovlossides load_settings
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
        self.test()
        # # perform test
        # if self.test_after_each_epoch == False:
        #     self.test()

        # plot
        # self.plot()