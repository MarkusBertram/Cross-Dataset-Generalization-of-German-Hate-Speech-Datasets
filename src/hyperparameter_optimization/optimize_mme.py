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
from src.model.MME_model import MME_model
import sys
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB

def get_target_dataset(target_labelled, target_train_split, validation_split, seed, truncation_length):

    # fetch labelled target dataset and split labelled target dataset into train and test
    labelled_target_dataset_features_train, labelled_target_dataset_features_test, labelled_target_dataset_labels_train, labelled_target_dataset_labels_test, abusive_ratio = fetch_dataset(
        target_labelled,
        labelled = True, 
        target = True,
        seed=seed,
        truncation_length=truncation_length,
        target_train_split=target_train_split
        )

    #labelled_target_features_train, _, labelled_target_labels_train, _ =  train_test_split(labelled_target_dataset_features_val, labelled_target_dataset_labels_val, test_size = (1-target_train_split), random_state = seed, stratify = labelled_target_dataset_labels)

    # further split train set into train and val
    labelled_target_features_train, labelled_target_features_val, labelled_target_labels_train, labelled_target_labels_val = train_test_split(labelled_target_dataset_features_train, labelled_target_dataset_labels_train, test_size = validation_split, random_state = seed, stratify = labelled_target_dataset_labels_train)
    
    return labelled_target_features_train, labelled_target_labels_train.float(), labelled_target_features_val, labelled_target_labels_val.float(), None, None, abusive_ratio

def preprocess(batch, tokenizer, truncation_length):
    batch = cleanTweets(batch)

    return pd.Series(tokenizer(batch, truncation=True, max_length=truncation_length, padding = "max_length",  return_token_type_ids = False))

def fetch_dataset(dataset_name, labelled = True, target = False, validation_split = None, seed = None, unlabelled_size = None, stratify_unlabelled = True, abusive_ratio = None, truncation_length = None, target_train_split = None):

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
    
    if labelled == True and target == False:
        #map neutral to 0 and abusive to 1
        label_df = dset_df["label"].map(label2id)
        labels_array = np.array(label_df.values.tolist())

        train_tokens_array, val_tokens_array, train_labels_array, val_labels_array = train_test_split(tokens_array, labels_array, test_size = validation_split, random_state = seed, stratify = labels_array)
        
        train_tokens_tensor =  torch.from_numpy(train_tokens_array)
        val_tokens_tensor =  torch.from_numpy(val_tokens_array)
        train_labels_tensor =  torch.from_numpy(train_labels_array).float()
        val_labels_tensor =  torch.from_numpy(val_labels_array).float()
        return train_tokens_tensor, val_tokens_tensor, train_labels_tensor, val_labels_tensor
    elif labelled == True and target == True:
        #map neutral to 0 and abusive to 1
        label_df = dset_df["label"].map(label2id)
        labels_array = np.array(label_df.values.tolist())

        train_tokens_array, test_tokens_array, train_labels_array, test_labels_array = train_test_split(tokens_array, labels_array, test_size = (1-target_train_split), random_state = seed, stratify = labels_array)
        
        train_tokens_tensor =  torch.from_numpy(train_tokens_array)
        test_tokens_tensor =  torch.from_numpy(test_tokens_array)
        train_labels_tensor =  torch.from_numpy(train_labels_array).float()
        test_labels_tensor =  torch.from_numpy(test_labels_array).float()
        return train_tokens_tensor, test_tokens_tensor, train_labels_tensor, test_labels_tensor, abusive_ratio
    else:
        #train_tokens_array, val_tokens_array = train_test_split(tokens_array, test_size = validation_split, random_state = seed)
        
        tokens_tensor =  torch.from_numpy(tokens_array)
        #val_tokens_tensor =  torch.from_numpy(val_tokens_array)
        #train_labels_tensor =  None
        #val_labels_tensor =  None
        return tokens_tensor#, val_tokens_tensor, train_labels_tensor, val_labels_tensor
    

def get_data_loaders(sources, 
        target_labelled,
        target_unlabelled,
        target_train_split,
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
        unlabelled_size = 0
        for source_name in sources:
            train_features, val_features, train_labels, val_labels = fetch_dataset(
                source_name,
                labelled = True,
                target = False,
                
                validation_split = validation_split,
                seed = seed,
                truncation_length=truncation_length)

            unlabelled_size += len(train_features) + len(val_features)
            
            source_features.append(train_features)
            source_labels.append(train_labels)
            val_features_list.append(val_features)
            val_labels_list.append(val_labels)

        # fetch labelled target train val test dataset
        labelled_target_features_train, labelled_target_labels_train, labelled_target_features_val, labelled_target_labels_val, _, _, abusive_ratio = get_target_dataset(
            target_labelled,
            target_train_split,
            validation_split,
            seed,
            truncation_length=truncation_length)
        
        val_features_list.append(labelled_target_features_val)
        val_labels_list.append(labelled_target_labels_val)
        
        combined_source_features = torch.cat(source_features)
        combined_source_labels = torch.cat(source_labels)

         # create labelled target dataloader
        labelled_target_dataset_train = TensorDataset(labelled_target_features_train, labelled_target_labels_train)
        sampler = BatchSampler(RandomSampler(labelled_target_dataset_train), batch_size=min(batch_size, len(labelled_target_dataset_train)), drop_last=True)
        labelled_target_dataloader = DataLoader(dataset=labelled_target_dataset_train, sampler = sampler, num_workers=num_workers)
         
        # concatenate datasets
        source_dataset = TensorDataset(combined_source_features, combined_source_labels)
        sampler = BatchSampler(RandomSampler(source_dataset), batch_size=min(batch_size, len(labelled_target_dataset_train)), drop_last=False)
        source_dataloader = DataLoader(dataset=source_dataset, sampler = sampler, num_workers=num_workers)

        del source_features
        del source_labels
        gc.collect()

        # fetch unlabelled target dataset
        unlabelled_target_dataset_features = fetch_dataset(
            target_unlabelled,
            labelled = False,
            target = True,
            seed = seed,
            unlabelled_size = unlabelled_size,
            stratify_unlabelled = stratify_unlabelled,
            abusive_ratio = abusive_ratio)
        unlabelled_target_dataset = TensorDataset(unlabelled_target_dataset_features)
        sampler = BatchSampler(RandomSampler(unlabelled_target_dataset), batch_size=2 * min(batch_size, len(labelled_target_dataset_train)), drop_last=True)
        unlabelled_target_dataloader = DataLoader(dataset=unlabelled_target_dataset, sampler = sampler, num_workers=num_workers)            
        del unlabelled_target_dataset_features
        gc.collect()

        # create test dataloader
        # use validation set as test set
        combined_val_features = torch.cat(val_features_list)
        combined_val_labels = torch.cat(val_labels_list)
        labelled_target_dataset_test = TensorDataset(combined_val_features, combined_val_labels)
        sampler = BatchSampler(RandomSampler(labelled_target_dataset_test), batch_size=min(batch_size, len(labelled_target_dataset_train)), drop_last=False)
        test_dataloader = DataLoader(dataset=labelled_target_dataset_test, sampler = sampler, num_workers=num_workers)               
        del labelled_target_dataset_test
        gc.collect()

        return source_dataloader, labelled_target_dataloader, unlabelled_target_dataloader, test_dataloader

def create_model(device, truncation_length):
    from src.model.feature_extractors import BERT_cnn
    feature_extractor = BERT_cnn(truncation_length)
    output_hidden_states = True
    
    from src.model.task_classifiers import DANN_task_classifier
    task_classifier = DANN_task_classifier()

    model = MME_model(feature_extractor, task_classifier, output_hidden_states).to(device)  
    
    return model
      
def train_mme(config, checkpoint_dir=None):
    seed= 123
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_labelled= "telegram_gold"
    target_unlabelled= "telegram_unlabeled"
    sources= [
            "germeval2018", 
            "germeval2019",
            "hasoc2019",
            "hasoc2020",
            "ihs_labelled",
            "covid_2021"
        ]
    target_train_split = 0.05
    validation_split= 0.1
    batch_size= 4
    num_workers= 2
    stratify_unlabelled= True

    truncation_length= 512
    
    model = create_model(device, truncation_length)

    model.to(device)
    
    source_dataloader, labelled_target_dataloader, unlabelled_target_dataloader, test_loader = get_data_loaders(
        sources = sources,
        target_labelled = target_labelled,
        target_unlabelled = target_unlabelled,
        target_train_split = target_train_split, 
        validation_split = validation_split, 
        batch_size = batch_size, 
        num_workers = num_workers,
        stratify_unlabelled = stratify_unlabelled,
        seed = seed,
        truncation_length = truncation_length
    )

    criterion = nn.BCEWithLogitsLoss()
    
    params = [{
            "params": model.feature_extractor.parameters(), "lr": config["lr"]
        }]
    optimizer_g = optim.Adam(
        params,
        betas=(config["beta1"],config["beta2"])
    )
    params = [{
        "params": model.task_classifier.parameters(), "lr": config["lr"]
    }]

    optimizer_f = optim.Adam(
        params,
        betas=(config["beta1"],config["beta2"])
    )
    
    step = 0

    if checkpoint_dir:
      path = os.path.join(checkpoint_dir, "checkpoint")
      checkpoint = torch.load(path)

      model.load_state_dict(checkpoint["model_state_dict"])
      optimizer_f.load_state_dict(checkpoint["optimizer_f_state_dict"])
      optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
      step = checkpoint["step"]
    
    for name, param in model.named_parameters():
        if "bert" in name:
            param.requires_grad = False

    eta = config["eta"]
    lamda = config["lamda"]

    data_iter_s = iter(source_dataloader)
    data_iter_t = iter(labelled_target_dataloader)
    data_iter_t_unl = iter(unlabelled_target_dataloader)
    len_train_source = len(source_dataloader)
    len_train_target = len(labelled_target_dataloader)
    len_train_target_semi = len(unlabelled_target_dataloader)

    while True:  # loop over the dataset multiple times
        model.train()
           
        if step % len_train_target == 0:
            data_iter_t = iter(labelled_target_dataloader)
        if step % len_train_target_semi == 0:
            data_iter_t_unl = iter(unlabelled_target_dataloader)
        if step % len_train_source == 0:
            data_iter_s = iter(source_dataloader)
        
        data_s = next(data_iter_s)
        data_t = next(data_iter_t)
        data_t_unl = next(data_iter_t_unl) 

        source_features = data_s[0][0].to(device)
        source_labels = data_s[1][0].to(device)
        labelled_target_features = data_t[0][0].to(device)
        labelled_target_labels = data_t[1][0].to(device)
        unlabelled_target_features = data_t_unl[0][0].to(device)
        
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        data = torch.cat((source_features, labelled_target_features), 0)
        target = torch.cat((source_labels, labelled_target_labels), 0)

        output = model(data)

        loss = criterion(output, target)
        loss.backward(retain_graph=True)

        optimizer_g.step()
        optimizer_f.step()

        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        out_t1 = model(unlabelled_target_features, reverse = True, eta = eta)
        # conditional entropy
        # https://github.com/VisionLearningGroup/SSDA_MME/blob/81c3a9c321f24204db7223405662d4d16b22b17c/utils/loss.py#L36
        loss_t = lamda * torch.mean(torch.sigmoid(out_t1)*F.logsigmoid(out_t1+1e-5))

        loss_t.backward()
        optimizer_f.step()
        optimizer_g.step()

        if step % len_train_source == len_train_source-1:
            model.eval()
            # Validation loss
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for i, data in enumerate(test_loader, 0):
                    inputs, labels = data
                    inputs, labels = inputs[0].to(device), labels[0].to(device)

                    outputs = model.inference(inputs)
                    predicted = torch.round(torch.sigmoid(outputs))

                    loss = criterion(predicted, labels)

                    val_loss += loss.cpu().numpy()
                    val_steps += 1
            avg_val_loss =  (val_loss / val_steps)

            # Here we save a checkpoint. It is automatically registered with
            # Ray Tune and will potentially be passed as the `checkpoint_dir`
            # parameter in future iterations.
            with tune.checkpoint_dir(step=step) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_f_state_dict": optimizer_f.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict()

                }, path)
            
            tune.report(loss=avg_val_loss)
        
        step += 1

if __name__ == "__main__":
    config_dict = {
        "lr": tune.loguniform(1e-5, 1e-4),
        "beta1": tune.loguniform(0.88, 0.999),
        "beta2": tune.loguniform(0.99, 0.9999),
        "eta": tune.loguniform(0.1, 100),
        "lamda": tune.loguniform(0.01, 1),
    }

    algo = TuneBOHB(
        metric = "loss",
        mode = "min"
        )

    bohb = HyperBandForBOHB(
        time_attr="training_iteration",
        metric = "loss",
        mode = "min",
        max_t=73530)
    
    result = tune.run(
        train_mme,
        name="mme",
        scheduler=bohb,
        search_alg = algo,
        resources_per_trial={
            "cpu": 2,
            "gpu": 1  # set this for GPUs
        },
        time_budget_s=82800,
        num_samples=-1,
        config=config_dict)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
