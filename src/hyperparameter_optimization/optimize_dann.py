from functools import partial
from pydoc import source_synopsis
from sre_parse import Tokenizer
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from filelock import FileLock
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import random
from src.utils.utils import fetch_import_module
import pandas as pd
from src.experiments.experiment_base import cleanTweets
from sklearn.model_selection import train_test_split
import gc
from src.utils.exp_utils import CustomConcatDataset
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from src.model import *
from ray.tune.schedulers import AsyncHyperBandScheduler
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.model.DANN_model import DANN_model

def get_target_dataset(target_labelled, train_split, validation_split, seed, truncation_length):

    # fetch labelled target dataset and split labelled target dataset into train and test
    labelled_target_dataset_features, labelled_target_dataset_labels, abusive_ratio = fetch_dataset(
        target_labelled,
        labelled = True, 
        target = True,
        truncation_length=truncation_length)

    labelled_target_features_train, _, labelled_target_labels_train, _ =  train_test_split(labelled_target_dataset_features, labelled_target_dataset_labels, test_size = (1-train_split), random_state = seed, stratify = labelled_target_dataset_labels)

    # further split train set into train and val
    _, labelled_target_features_val, _, labelled_target_labels_val = train_test_split(labelled_target_features_train, labelled_target_labels_train, test_size = validation_split, random_state = seed, stratify = labelled_target_labels_train)
    
    return _, _, torch.from_numpy(labelled_target_features_val), torch.from_numpy(labelled_target_labels_val).float(), _, _, abusive_ratio

def preprocess(batch, tokenizer, truncation_length):
    batch = cleanTweets(batch)

    return pd.Series(tokenizer(batch, truncation=True, max_length=truncation_length, padding = "max_length",  return_token_type_ids = False))

def fetch_dataset(dataset_name, labelled = True, target = False, return_val = False, validation_split = None, seed = None, unlabelled_size = None, stratify_unlabelled = True, abusive_ratio = None, truncation_length = None):

    ###### fetch datasets
    label2id = {"neutral": 0, "abusive":1}
    # import dataset pipeline
    dset_module = fetch_import_module(dataset_name)
    # execute get_data_binary in pipeline
    if labelled == True:
        dset_list_of_dicts = dset_module.get_data_binary()
    elif labelled == False:
        dset_list_of_dicts = dset_module.get_data_binary(unlabelled_size, stratify = stratify_unlabelled, abusive_ratio = abusive_ratio)
    # convert list to dataframe
    dset_df = pd.DataFrame(dset_list_of_dicts)
    if labelled == True and target == True:
        abusive_ratio = dset_df["label"].value_counts(normalize = True)["abusive"]
    
    # tokenize each row in dataframe
    tokenizer = AutoTokenizer.from_pretrained('deepset/gbert-base')
    tokens_df = dset_df.apply(lambda row: preprocess(row.text, tokenizer, truncation_length), axis='columns', result_type='expand')

    tokens_array = np.array(tokens_df[["input_ids", "attention_mask"]].values.tolist())
    
    if return_val:
        # map neutral to 0 and abusive to 1
        label_df = dset_df["label"].map(label2id)
        labels_array = np.array(label_df.values.tolist())

        train_tokens_array, val_tokens_array, train_labels_array, val_labels_array = train_test_split(tokens_array, labels_array, test_size = validation_split, random_state = seed, stratify = labels_array)
        
        train_tokens_tensor =  torch.from_numpy(train_tokens_array)
        val_tokens_tensor =  torch.from_numpy(val_tokens_array)
        train_labels_tensor =  torch.from_numpy(train_labels_array).float()
        val_labels_tensor =  torch.from_numpy(val_labels_array).float()
        
        return train_tokens_tensor, val_tokens_tensor, train_labels_tensor, val_labels_tensor
        
    else:

        if labelled == True:
            # map neutral to 0 and abusive to 1
            label_df = dset_df["label"].map(label2id)
            labels_array = np.array(label_df.values.tolist())
            #labels_tensor = torch.from_numpy(labels_array).float()
        else:
            labels_array = None

        if abusive_ratio is not None:
            return tokens_array, labels_array, abusive_ratio
        else:
            return tokens_array, labels_array, abusive_ratio

def get_data_loaders(sources, 
        target_labelled,
        target_unlabelled,
        train_split,
        validation_split,
        batch_size,
        num_workers,
        stratify_unlabelled,
        seed,
        truncation_length):

    # https://docs.ray.io/en/releases-1.11.0/tune/tutorials/tune-pytorch-cifar.html
    with FileLock(os.path.expanduser("~/.data.lock")):
        # fetch source datasets
        source_features = []
        source_labels = []

        val_features_list = []
        val_labels_list = []

        for source_name in sources:
            train_features, val_features, train_labels, val_labels = fetch_dataset(
                source_name,
                labelled = True,
                target = False,
                return_val = True,
                validation_split = validation_split,
                seed = seed,
                truncation_length=truncation_length)

            source_features.append(train_features)
            source_labels.append(train_labels)

            val_features_list.append(val_features)
            val_labels_list.append(val_labels)
        
        # fetch labelled target train dataset
        _, _, labelled_target_features_val, labelled_target_labels_val, _, _, abusive_ratio = get_target_dataset(
            target_labelled,
            train_split,
            validation_split,
            seed,
            truncation_length=truncation_length)
        
        source_features.append(labelled_target_features_val)
        source_labels.append(labelled_target_labels_val)

        combined_source_features = torch.cat(source_features)
        combined_source_labels = torch.cat(source_labels)
        
        # concatenate datasets
        source_dataset = TensorDataset(combined_source_features, combined_source_labels)
        
        del source_features
        del source_labels
        gc.collect()

        # fetch unlabelled target dataset
        unlabelled_size = len(source_dataset)
        unlabelled_target_dataset_features, _, _ = fetch_dataset(
            target_unlabelled,
            labelled = False,
            target = True,
            unlabelled_size = unlabelled_size,
            stratify_unlabelled = stratify_unlabelled,
            abusive_ratio = abusive_ratio)
        
        # combine source dataset and unlabelled target dataset into one dataset
        concatenated_train_dataset = CustomConcatDataset(source_dataset, unlabelled_target_dataset_features)
        sampler = BatchSampler(RandomSampler(concatenated_train_dataset), batch_size=batch_size, drop_last=False)
        train_dataloader = DataLoader(dataset=concatenated_train_dataset, sampler = sampler, num_workers=num_workers)            
        #train_dataloader = DataLoader(concatenated_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        del concatenated_train_dataset
        del unlabelled_target_dataset_features
        gc.collect()

        # create test dataloader
        # use validation set as test set
        combined_val_features = torch.cat(val_features_list)
        combined_val_labels = torch.cat(val_labels_list)
        labelled_target_dataset_test = TensorDataset(combined_val_features, combined_val_labels)
        sampler = BatchSampler(RandomSampler(labelled_target_dataset_test), batch_size=batch_size, drop_last=False)
        test_dataloader = DataLoader(dataset=labelled_target_dataset_test, sampler = sampler, num_workers=num_workers)               
        del labelled_target_dataset_test
        gc.collect()

        return train_dataloader, test_dataloader

def create_model(device, truncation_length):
    from src.model.feature_extractors import BERT_cnn
    feature_extractor = BERT_cnn(truncation_length)
    output_hidden_states = True
    
    from src.model.task_classifiers import DANN_task_classifier
    task_classifier = DANN_task_classifier()
     
    from src.model.domain_classifiers import DANN_domain_classifier
    domain_classifier = DANN_domain_classifier()

    model = DANN_model(feature_extractor, task_classifier, domain_classifier, output_hidden_states).to(device)  
    
    return model

def train(model, optimizer, train_loader, epoch, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.BCEWithLogitsLoss()

    len_dataloader = len(train_loader)

    for i, data in enumerate(train_loader, 0):
        # zero the parameter gradients
        optimizer.zero_grad()

        # get the inputs; data is a list of [inputs, labels]
        source_batch, unlabelled_target_features = data

        p = float(i + epoch * len_dataloader) / epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        source_features = source_batch[0][0].to(device)
        source_labels = source_batch[1][0].to(device)

        batch_size = len(source_labels)

        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.float().to(device)

        class_output, domain_output = model(input_data=source_features, alpha=alpha)

        loss_s_label = criterion(class_output, source_labels)
        loss_s_domain = criterion(domain_output, domain_label)

        # training model using target data
        unlabelled_target_features = unlabelled_target_features[0].to(device)

        batch_size = len(unlabelled_target_features)

        domain_label = torch.ones(batch_size)
        domain_label = domain_label.float().to(device)

        _, domain_output = model(input_data=unlabelled_target_features, alpha=alpha)
        loss_t_domain = criterion(domain_output, domain_label)
        loss = loss_t_domain + loss_s_domain + loss_s_label

        loss.backward()
        optimizer.step()

def test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    # Validation loss
    val_loss = 0.0
    val_steps = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            inputs, labels = inputs[0].to(device), labels[0].to(device)

            outputs = model.inference(inputs)
            predicted = torch.round(torch.sigmoid(outputs)).int()

            loss = criterion(predicted, labels)
            val_loss += loss.cpu().numpy()
            val_steps += 1

    return (val_loss / val_steps)

def train_dann(config, checkpoint_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    epochs = config["epochs"]
    
    model = create_model(device, config["truncation_length"])

    model.to(device)
    
    optimizer = optim.Adam(
        model.parameters(), lr=config["lr"].sample())

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_loader, test_loader = get_data_loaders(
        sources = config["sources"],
        target_labelled = config["target_labelled"],
        target_unlabelled = config["target_unlabelled"],
        train_split = config["train_split"], 
        validation_split = config["validation_split"], 
        batch_size = config["batch_size"], 
        num_workers = config["num_workers"],
        stratify_unlabelled = config["stratify_unlabelled"],
        seed = config["seed"],
        truncation_length = config["truncation_length"]
    )
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        train(model, optimizer, train_loader, epoch, epochs)
        val_loss = test(model, test_loader)

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be passed as the `checkpoint_dir`
        # parameter in future iterations.
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path)

        #tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
        tune.report(loss=val_loss)
    print("Finished Training")

if __name__ == "__main__":
    sched = ASHAScheduler()

    config_dict = {
        "seed": 123,
        "lr": tune.loguniform(1e-4, 1e-2),
        "target_labelled": "telegram_gold",
        "target_unlabelled": "telegram_unlabeled",
        "sources": [
                "germeval2018", 
                "germeval2019"
            ],
        "train_split": 0.05,
        "validation_split": 0.1,
        "batch_size": 8,
        "num_workers": 2,
        "epochs": 10,
        "stratify_unlabelled": True,
        "truncation_length": 512
    }

    result = tune.run(
        train_dann,
        metric="loss",
        mode="min",
        name="dann",
        scheduler=sched,
        resources_per_trial={
            "cpu": 2,
            "gpu": 2  # set this for GPUs
        },
        num_samples=10,
        config=config_dict)

    print("Best config is:", result.best_config)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
