from typing import Dict, List, Union

# python
import datetime
import os
import json
import numpy as np
from model.DANN_model import DANN_model     
from utils.utils import (fetch_import_module, get_tweet_timestamp,
                         preprocess_text, print_data_example,
                         separate_text_by_classes)

import pandas as pd
import yaml
from torch.utils.tensorboard.writer import SummaryWriter
from datasets import Dataset, Features, ClassLabel
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
#from .helpers.measures import accuracy, auroc, f1

from experiment_base import experiment_base

class experiment_DANN(experiment_base):
    def __init__(
        self,
        basic_settings: Dict,
        exp_settings: Dict,
        #log_path: str,
        #writer: SummaryWriter,
    ):
        super(experiment_DANN, self).__init__(basic_settings, exp_settings)#, log_path, writer)
        #self.log_path = log_path
        #self.writer = writer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #basic_settings.update(exp_settings)
        self.current_experiment = exp_settings

        #self.load_settings()
        
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        

    # overrides train
    def train(self, train_loader, val_loader, optimizer, criterion, device, **kwargs):
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
        # verbose = kwargs.get("verbose", 1)

        lr_sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            factor=0.1,
            patience=int(self.epochs * 0.05),
            min_lr=1e-7,
        )

        for epoch in range(1, self.epochs + 1):

            train_loss = 0
            train_acc = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                if len(data) > 1:
                    self.model.train()
                    data, target = data.to(device).float(), target.to(device).long()

                    optimizer.zero_grad(set_to_none=True)
                    yhat = self.model(data).to(device)
                    loss = criterion(yhat, target)
                    try:
                        train_loss += loss.item()
                    except:
                        print("loss item skipped loss")
                    train_acc += torch.sum(torch.argmax(yhat, dim=1) == target).item()

                    loss.backward()
                    optimizer.step()
                else:
                    pass

            avg_train_loss = train_loss / len(train_loader)
            avg_train_acc = train_acc / len(train_loader.dataset)

            if epoch % 1 == 0:
                if validation:
                    val_loss = 0
                    val_acc = 0
                    self.model.eval()  # prep self.model for evaluation
                    with torch.no_grad():
                        for vdata, vtarget in val_loader:
                            vdata, vtarget = (
                                vdata.to(device).float(),
                                vtarget.to(device).long(),
                            )
                            voutput = self.model(vdata)
                            vloss = criterion(voutput, vtarget)
                            val_loss += vloss.item()
                            val_acc += torch.sum(
                                torch.argmax(voutput, dim=1) == vtarget
                            ).item()

                    avg_val_loss = val_loss / len(self.val_loader)
                    avg_val_acc = val_acc / len(self.val_loader.dataset)

                    early_stopping(avg_val_loss, self.model)
                    if kwargs.get("lr_sheduler", True):
                        lr_sheduler.step(avg_val_loss)

                    verbosity(
                        f"Val_loss: {avg_val_loss:.4f} Val_acc : {100*avg_val_acc:.2f}",
                        self.verbose,
                        epoch,
                    )

                    if early_stopping.early_stop:
                        print(
                            f"Early stopping epoch {epoch} , avg train_loss {avg_train_loss}, avg val loss {avg_val_loss}"
                        )
                        break

            verbosity(
                f"Train_loss: {avg_train_loss:.4f} Train_acc : {100*avg_train_acc:.2f}",
                self.verbose,
                epoch,
            )

        self.avg_train_loss_hist = avg_train_loss
        self.avg_val_loss_hist = avg_val_loss
        self.avg_train_acc_hist = avg_train_acc
        self.avg_val_loss_hist = avg_val_acc

    # overrides test
    @torch.no_grad()
    def test(self):
        """test [computes loss of the test set]
        [extended_summary]
        Returns:
            [type]: [description]
        """
        test_loss = 0
        test_acc = 0
        self.model.eval()
        for (t_data, t_target) in self.test_loader:
            t_data, t_target = (
                t_data.to(self.device).float(),
                t_target.to(self.device).long(),
            )

            t_output = self.model(t_data)
            t_output.to(self.device).long()
            t_loss = self.criterion(t_output, t_target)
            test_loss += t_loss
            test_acc += torch.sum(torch.argmax(t_output, dim=1) == t_target).item()

        self.avg_test_acc = test_acc / len(self.test_loader.dataset)
        self.avg_test_loss = test_loss.to("cpu").detach().numpy() / len(
            self.test_loader
        )  # return avg testloss

    def create_optimizer(self) -> None:
        self.optimizer = optim.Adam(
            self.model.parameters(),
            weight_decay=self.weight_decay,
            lr=self.lr,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )

    def create_criterion(self) -> None:
        self.criterion = nn.CrossEntropyLoss()

    # overrides load_settings
    def load_exp_settings(self) -> None:
        self.exp_name = self.current_experiment.get("exp_name", "standard_name")   
        self.feature_extractor = self.current_experiment.get("feature_extractor", "BERT_cls")
        self.task_classifier = self.current_experiment.get("task_classifier", "tc1")
        self.domain_classifier = self.current_experiment.get("domain_classifier", "dc1")
        

    def create_model(self):

        if self.feature_extractor == "BERT_cls":
            from .model.feature_extractors import BERT_cls
            #import .model.feature_extractors
            feature_extractor = BERT_cls()
            output_hidden_states = False
        elif self.feature_extractor == "BERT_cnn":
            from .model.feature_extractors import BERT_cnn
            feature_extractor = BERT_cnn()
            output_hidden_states = True
        else:
            print("error, can't find this feature extractor. please specify bert_cls or bert_cnn in experiment settings.")
        
        if self.task_classifier == "tc1":
            from .model.task_classifiers import task_classifier1
            task_classifier = task_classifier1()
            
        if self.domain_classifier == "dc1":
            from .model.domain_classifiers import domain_classifier1
            domain_classifier = domain_classifier1()

        self.model = DANN_model(feature_extractor, task_classifier, domain_classifier, output_hidden_states)

    def create_dataloader(self):
        # fetch source datasets
        source_datasets = []
        for source_name in self.sources:
            source_datasets.append(self.fetch_dataset(source_name, labelled = True, target = False))
        
        # fetch labelled target dataset
        labelled_target_dataset = self.fetch_dataset(self.target_labelled, labelled = True, target = True)

        labelled_target_dataset = labelled_target_dataset.train_test_split(test_size = self.test_split, stratify_by_column = "label")
        labelled_target_dataset_train = labelled_target_dataset["train"]
        labelled_target_dataset_test = labelled_target_dataset["test"]
        del labelled_target_dataset
        gc.collect()

        source_datasets.append(labelled_target_dataset_train)

        # create source dataloader
        self.dataloader_source = DataLoader(dataset=source_datasets, batch_size = self.batch_size, shuffle=True, num_workers=self.num_workers)
        del source_datasets
        del labelled_target_dataset_train
        gc.collect()

        # fetch unlabelled target dataset and create dataloader
        unlabelled_target_dataset = self.fetch_dataset(self.target_unlabelled, labelled = False, target = True)
        self.dataloader_unlabelled_target = DataLoader(dataset=unlabelled_target_dataset, batch_size = self.batch_size, shuffle=True, num_workers=self.num_workers)
        del unlabelled_target_dataset
        gc.collect()

        # create test dataloader
        self.test_dataloader = DataLoader(dataset=labelled_target_dataset_test, batch_size = self.batch_size, shuffle=True, num_workers=self.num_workers)               
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

        # , train_loader, val_loader, optimizer, criterion, device
        self.train(
            self.train_loader,
            self.val_loader,
            self.optimizer,
            self.criterion,
            self.device,
        )
        self.test()

        self.current_oracle_step += 1
        if len(self.pool_loader) > 0:
            (
                pool_predictions,
                pool_labels_list,
            ) = self.pool_predictions(self.pool_loader)

            self.sampler(
                self.data_manager,
                number_samples=self.oracle_stepsize,
                net=self.model,
                predictions=pool_predictions,
            )

            test_predictions, test_labels = self.pool_predictions(self.test_loader)

            test_accuracy = accuracy(test_labels, test_predictions)
            f1_score = f1(test_labels, test_predictions)

            dict_to_add = {
                "test_loss": self.avg_test_loss,
                "train_loss": self.avg_train_loss_hist,
                "test_accuracy": test_accuracy,
                "train_accuracy": self.avg_train_acc_hist,
                "f1": f1_score,
            }

            print(dict_to_add)
            # if self.metric.lower() == "auroc":
            #     auroc_score = auroc(self.data_manager, oracle_s)

            #     dict_to_add = {"auroc": auroc_score}

            self.data_manager.add_log(
                writer=self.writer,
                oracle=self.oracle,
                dataset=self.iD,
                metric=self.metric,
                log_dict=dict_to_add,
                ood_ratio=self.OOD_ratio,
                exp_name=self.exp_name,
            )