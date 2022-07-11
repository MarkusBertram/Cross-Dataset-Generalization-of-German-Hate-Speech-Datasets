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
from pandarallel import pandarallel

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
    ):
        super(experiment_DANN, self).__init__(basic_settings, exp_settings)#, log_path, writer)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.current_experiment = exp_settings
        
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        

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

        for epoch in range(1, self.epochs + 1):
            len_dataloader = len(self.train_dataloader)
            print(len_dataloader)
            sys.exit(0)
            # data_source_iter = iter(self.dataloader_source)
            # data_target_iter = iter(self.dataloader_unlabelled_target)

            # i = 0
            # while i < len_dataloader:
            for i, (source_batch, unlabelled_target_features) in enumerate(self.train_dataloader):

                print(i)
                source_features = source_batch[0].squeeze().to(self.device)
                source_labels = source_batch[1].squeeze().to(self.device)
                unlabelled_target_features = unlabelled_target_features.squeeze().to(self.device)
                
                p = float(i + epoch * len_dataloader) / self.epochs / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                # training model using source data
                data_source = data_source_iter.next()
                s_features, s_label = data_source

                self.model.zero_grad()
                batch_size = len(s_label)

                #input_features = torch.FloatTensor(batch_size, 3, image_size, image_size)
                class_label = torch.LongTensor(batch_size)
                domain_label = torch.zeros(batch_size)
                domain_label = domain_label.long()

                s_features = s_features.to(self.device)
                s_label = s_label.to(self.device)
                input_features = input_features.to(self.device)
                class_label = class_label.to(self.device)
                domain_label = domain_label.to(self.device)


                input_features.resize_as_(s_features).copy_(s_features)
                class_label.resize_as_(s_label).copy_(s_label)

                class_output, domain_output = self.model(input_data=input_features, alpha=alpha)
                err_s_label = loss_class(class_output, class_label)
                err_s_domain = loss_domain(domain_output, domain_label)

                # training model using target data
                data_target = data_target_iter.next()
                t_features, _ = data_target

                batch_size = len(t_features)

                #input_features = torch.FloatTensor(batch_size, 3, image_size, image_size)
                domain_label = torch.ones(batch_size)
                domain_label = domain_label.long()

                t_features = t_features.to(self.device)
                input_features = input_features.to(self.device)
                domain_label = domain_label.to(self.device)

                input_features.resize_as_(t_features).copy_(t_features)

                _, domain_output = self.model(input_data=input_features, alpha=alpha)
                err_t_domain = loss_domain(domain_output, domain_label)
                err = err_t_domain + err_s_domain + err_s_label
                err.backward()
                self.optimizer.step()

                i += 1
        #     train_loss = 0
        #     train_acc = 0
        #     for batch_idx, (data, target) in enumerate(train_loader):
        #         if len(data) > 1:
        #             self.model.train()
        #             data, target = data.to(device).float(), target.to(device).long()

        #             optimizer.zero_grad(set_to_none=True)
        #             yhat = self.model(data).to(device)
        #             loss = criterion(yhat, target)
        #             try:
        #                 train_loss += loss.item()
        #             except:
        #                 print("loss item skipped loss")
        #             train_acc += torch.sum(torch.argmax(yhat, dim=1) == target).item()

        #             loss.backward()
        #             optimizer.step()
        #         else:
        #             pass

        #     avg_train_loss = train_loss / len(train_loader)
        #     avg_train_acc = train_acc / len(train_loader.dataset)

        #     if epoch % 1 == 0:
        #         if validation:
        #             val_loss = 0
        #             val_acc = 0
        #             self.model.eval()  # prep self.model for evaluation
        #             with torch.no_grad():
        #                 for vdata, vtarget in val_loader:
        #                     vdata, vtarget = (
        #                         vdata.to(device).float(),
        #                         vtarget.to(device).long(),
        #                     )
        #                     voutput = self.model(vdata)
        #                     vloss = criterion(voutput, vtarget)
        #                     val_loss += vloss.item()
        #                     val_acc += torch.sum(
        #                         torch.argmax(voutput, dim=1) == vtarget
        #                     ).item()

        #             avg_val_loss = val_loss / len(self.val_loader)
        #             avg_val_acc = val_acc / len(self.val_loader.dataset)

        #             early_stopping(avg_val_loss, self.model)
        #             if kwargs.get("lr_sheduler", True):
        #                 lr_sheduler.step(avg_val_loss)

        #             verbosity(
        #                 f"Val_loss: {avg_val_loss:.4f} Val_acc : {100*avg_val_acc:.2f}",
        #                 self.verbose,
        #                 epoch,
        #             )

        #             if early_stopping.early_stop:
        #                 print(
        #                     f"Early stopping epoch {epoch} , avg train_loss {avg_train_loss}, avg val loss {avg_val_loss}"
        #                 )
        #                 break

        #     verbosity(
        #         f"Train_loss: {avg_train_loss:.4f} Train_acc : {100*avg_train_acc:.2f}",
        #         self.verbose,
        #         epoch,
        #     )

        # self.avg_train_loss_hist = avg_train_loss
        # self.avg_val_loss_hist = avg_val_loss
        # self.avg_train_acc_hist = avg_train_acc
        # self.avg_val_loss_hist = avg_val_acc

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
            print("error, can't find this feature extractor. please specify bert_cls or bert_cnn in experiment settings.")
        
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

        # plot
        # self.plot()