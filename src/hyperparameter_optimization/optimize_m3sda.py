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
from src.model.M3SDA_model import M3SDA_model
import sys
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB
from collections import defaultdict

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
        source_datasets = []

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
            
            source_datasets.append(TensorDataset(train_features, train_labels))
            val_features_list.append(val_features)
            val_labels_list.append(val_labels)

        # fetch labelled target train val test dataset
        labelled_target_features_train, labelled_target_labels_train, labelled_target_features_val, labelled_target_labels_val, _, _, abusive_ratio = get_target_dataset(
            target_labelled,
            target_train_split,
            validation_split,
            seed,
            truncation_length=truncation_length)
        
        source_datasets.append(TensorDataset(labelled_target_features_train, labelled_target_labels_train))
        val_features_list.append(labelled_target_features_val)
        val_labels_list.append(labelled_target_labels_val)
         
        # concatenate datasets
        source_dataloader_list = []
        for source_dataset in source_datasets:
            sampler = BatchSampler(RandomSampler(source_dataset), batch_size=batch_size, drop_last=True)
            dataloader = DataLoader(dataset=source_dataset, sampler = sampler, num_workers=num_workers)
            source_dataloader_list.append(dataloader)

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
        sampler = BatchSampler(RandomSampler(unlabelled_target_dataset), batch_size=batch_size, drop_last=True)
        unlabelled_target_dataloader = DataLoader(dataset=unlabelled_target_dataset, sampler = sampler, num_workers=num_workers)            
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

        return source_dataloader_list, unlabelled_target_dataloader, test_dataloader

def create_model(device, truncation_length, num_sources):
    from src.model.feature_extractors import BERT_cnn
    feature_extractor = BERT_cnn(truncation_length)
    output_hidden_states = True

    from src.model.task_classifiers import DANN_task_classifier
    task_classifier = DANN_task_classifier()

    model = M3SDA_model(feature_extractor, task_classifier, output_hidden_states, num_sources).to(device)  
    
    return model
      
def train_m3sda(config, checkpoint_dir=None):
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
    
    source_dataloader_list, unlabelled_target_dataloader, test_loader = get_data_loaders(
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

    model = create_model(device, truncation_length, len(source_dataloader_list))

    model.to(device)
    
    N = len(source_dataloader_list)
    epoch = 0
    epochs = 10
    # setting up optimizers
    extractor_optimizer = optim.Adam(
        model.feature_extractor.parameters(),
        lr = config["lr"],
        betas=(config["beta1"],config["beta2"])
    )
    predictor_optimizers = []
    for i in range(N):
        predictor_optimizers.append(optim.Adam(list(model.task_classifiers[i][0].parameters()) + list(model.task_classifiers[i][1].parameters()), lr = config["lr"]))

    if checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint")
        checkpoint = torch.load(path)

        model.load_state_dict(checkpoint["model_state_dict"])
        extractor_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]

        for i in range(N):
            predictor_optimizers[i].load_state_dict(checkpoint[f"predictor_optimizer_{i}"])
    
    for name, param in model.named_parameters():
        if "bert" in name:
            param.requires_grad = False

    loss_extractor = nn.BCEWithLogitsLoss()
    loss_l1 = nn.L1Loss()
    bce_loss = nn.BCEWithLogitsLoss()

    lambd = config["lambd"]
    discrepancy_weight = config["discrepancy_weight"]

    while True:  # loop over the dataset multiple times
        model.train()

        source_ac = {}
        for i in range(N):
            source_ac[i] = defaultdict(int)

        train_dataloader = multi_data_loader(source_dataloader_list, unlabelled_target_dataloader)

        for batch_index, (source_batches, unlabelled_target_batch) in enumerate(train_dataloader):

            loss_cls = 0
            
            ######### train extractor and source classifier, Step i.
            # First term of equation (2)
            for index, batch in enumerate(source_batches):
                features = batch[0][0]
                labels = batch[1][0]
                
                features = features.to(device)
                labels = labels.to(device)

                pred1, pred2 = model(features, index)

                source_ac[index]['c1'] += torch.sum(torch.round(pred1) == labels).item()
                source_ac[index]['c2'] += torch.sum(torch.round(pred2) == labels).item()
                source_ac[index]['count'] += len(labels)

                loss1 = loss_extractor(pred1, labels)
                loss2 = loss_extractor(pred2, labels)

                loss_cls += loss1 + loss2
                            
            # Moment Distance Loss
            # Second Term of equation (2)
            m1_loss = 0
            m2_loss = 0

            for k in range(1,3):
                for i_index, batch in enumerate(source_batches):
                    
                    unlabelled_target_features = unlabelled_target_batch[0][0].to(device)
                    
                    features = batch[0][0]
                    labels = batch[1][0]
                    
                    features = features.to(device)
                    labels = labels.to(device)

                    src_feature = model(features, output_only_features = True)
                    tar_feature = model(unlabelled_target_features, output_only_features = True)

                    
                    e_src = torch.mean(src_feature**k, dim=0)
                    e_tar = torch.mean(tar_feature**k, dim=0)
                    m1_dist = e_src.dist(e_tar)
                    m1_loss += m1_dist

                    for j_index, other_batch in enumerate(source_batches[i_index+1:]):
                        other_x = other_batch[0][0]
                        other_y = other_batch[1][0]

                        other_x = other_x.to(device)
                        other_y = other_y.to(device)

                        other_feature = model(other_x, output_only_features = True)

                        e_other = torch.mean(other_feature**k, dim=0)
                        m2_dist = e_src.dist(e_other)
                        m2_loss += m2_dist

            loss_m =  (epochs-epoch)/epochs * (m1_loss/N + m2_loss/N/(N-1)*2) * lambd

            loss = loss_cls + loss_m
            
            extractor_optimizer.zero_grad(set_to_none = True)

            for i in range(N):
                predictor_optimizers[i].zero_grad(set_to_none = True)
            
            loss.backward()

            extractor_optimizer.step()
            
            for i in range(N):
                predictor_optimizers[i].step()
                predictor_optimizers[i].zero_grad()
                
            extractor_optimizer.zero_grad()

            ######### Step ii.

            unlabelled_target_features = unlabelled_target_batch[0][0].to(device)
                
            loss = 0
            d_loss = 0
            c_loss = 0

            for index, batch in enumerate(source_batches):
                features = batch[0][0]
                labels = batch[1][0]
                
                features = features.to(device)
                labels = labels.to(device)

                src_pred1, src_pred2 = model(features, index=index)
                # First term of equation (3)
                c_loss += loss_extractor(src_pred1, labels) + loss_extractor(src_pred2, labels)


                tgt_pred1, tgt_pred2 = model(unlabelled_target_features, index=index)
                
                combine1 = (torch.sigmoid(tgt_pred1) + torch.sigmoid(tgt_pred2))/2
                # Second term of equation (3)
                d_loss += loss_l1(tgt_pred1, tgt_pred2)

                for index_2, o_batch in enumerate(source_batches[index+1:]):
                    
                    pred_2_c1, pred_2_c2 = model(unlabelled_target_features, index = index_2+index)
                    
                    combine2 = (torch.sigmoid(pred_2_c1) + torch.sigmoid(pred_2_c2))/2

                    d_loss += loss_l1(combine1, combine2) * discrepancy_weight
            # Equation (3)
            loss = c_loss - d_loss 
            
            loss.backward()
            extractor_optimizer.zero_grad(set_to_none = True)

            for i in range(N):
                predictor_optimizers[i].zero_grad()
            
            for i in range(N):
                predictor_optimizers[i].step()
            
            for i in range(N):
                predictor_optimizers[i].zero_grad()
            
            ########## Step iii.

            all_dis = 0

            for i in range(3):
                discrepency_loss = 0
                tar_feature = model(unlabelled_target_features, output_only_features = True)

                for index, _ in enumerate(source_batches):
                    
                    pred_c1, pred_c2 = model(tar_feature, index = index, feature_extractor_input = True)
                    
                    combine1 = (torch.sigmoid(pred_c1) + torch.sigmoid(pred_c2))/2
                    # Equation (4)
                    discrepency_loss += loss_l1(pred_c1, pred_c2)

                    for index2, _ in enumerate(source_batches[index+1:]):
                        
                        pred_2_c1, pred_2_c2 = model(tar_feature, index = index2+index, feature_extractor_input = True)

                        combine2 = (torch.sigmoid(pred_2_c1) + torch.sigmoid(pred_2_c2))/2

                        discrepency_loss += loss_l1(combine1, combine2) * discrepancy_weight

                all_dis += discrepency_loss.item()

                extractor_optimizer.zero_grad(set_to_none = True)

                for i in range(N):
                    predictor_optimizers[i].zero_grad(set_to_none = True)

                discrepency_loss.backward()

                extractor_optimizer.step()
                extractor_optimizer.zero_grad(set_to_none = True)
                
                for i in range(N):
                    predictor_optimizers[i].zero_grad(set_to_none = True)

        # set weights to highest source accuracy
        for i in range(N):
            c1_acc = source_ac[i]['c1']/source_ac[i]['count']
            c2_acc = source_ac[i]['c2']/source_ac[i]['count']
            model.weights[i] = max(c1_acc, c2_acc)


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

                loss = bce_loss(predicted, labels)

                val_loss += loss.cpu().numpy()
                val_steps += 1
        avg_val_loss =  (val_loss / val_steps)

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be passed as the `checkpoint_dir`
        # parameter in future iterations.
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")

            torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "extractor_optimizer_state_dict": extractor_optimizer.state_dict()
            }, path)

            for i in range(N):
                torch.save({
                    f"predictor_optimizer_{i}": predictor_optimizers[i].state_dict()
                    }, path)
        
        tune.report(loss=avg_val_loss)
        
if __name__ == "__main__":
    config_dict = {
        "lr": tune.uniform(1e-5, 1e-4),
        "beta1": tune.loguniform(0.88, 0.999),
        "beta2": tune.loguniform(0.99, 0.9999),
        "lambd": tune.uniform(0.1, 1),
        "discrepancy_weight": tune.loguniform(1e-2, 1)
    }

    algo = TuneBOHB(
        metric = "loss",
        mode = "min"
        )

    bohb = HyperBandForBOHB(
        time_attr="training_iteration",
        metric = "loss",
        mode = "min",
        max_t = 10)
    
    result = tune.run(
        train_m3sda,
        name="m3sda",
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
