from functools import partial
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

def preprocess(batch):
    batch = cleanTweets(batch)

    return pd.Series(tokenizer(batch, truncation=True, max_length=truncation_length, padding = "max_length",  return_token_type_ids = False))

def fetch_dataset(dataset_name, labelled = True, target = False, return_val = False):

    ###### fetch datasets
    label2id = {"neutral": 0, "abusive":1}
    # import dataset pipeline
    dset_module = fetch_import_module(dataset_name)
    # execute get_data_binary in pipeline
    if labelled == True and target == False:
        dset_list_of_dicts = dset_module.get_data_binary()
    elif labelled == True and target == True:
        dset_list_of_dicts = dset_module.get_data_binary()
    elif labelled == False and target == True:
        dset_list_of_dicts = dset_module.get_data_binary(unlabelled_size, stratify = stratify_unlabelled, abusive_ratio = abusive_ratio)
    # convert list to dataframe
    dset_df = pd.DataFrame(dset_list_of_dicts)
    if labelled == True and target == True:
        abusive_ratio = dset_df["label"].value_counts(normalize = True)["abusive"]
    
    # tokenize each row in dataframe
    tokens_df = dset_df.apply(lambda row: preprocess(row.text), axis='columns', result_type='expand')

    tokens_array = np.array(tokens_df[["input_ids", "attention_mask"]].values.tolist())
    
    if return_val:

        # map neutral to 0 and abusive to 1
        label_df = dset_df["label"].map(label2id)
        labels_array = np.array(label_df.values.tolist())

        train_tokens_array, val_tokens_array, train_labels_array, val_labels_array = train_test_split(tokens_array, labels_array, test_size = self.validation_split, random_state = self.seed, stratify = labels_array)
        
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

        return tokens_array, labels_array

def get_data_loaders(dset_type):
    # https://docs.ray.io/en/releases-1.11.0/tune/tutorials/tune-pytorch-cifar.html
    with FileLock(os.path.expanduser("~/.data.lock")):
        if dset_type == "unsupervised":
            
            # fetch source datasets
            source_features = []
            source_labels = []

            val_features = []
            val_labels = []

            for source_name in sources:
                train_features, val_features, train_labels, val_labels = fetch_dataset(source_name, labelled = True, target = False, return_val = True)

                source_features.append(train_features)
                source_labels.append(train_labels)

                val_features.append(val_features)
                val_labels.append(val_labels)
            
            # fetch labelled target dataset
            labelled_target_features_train, labelled_target_labels_train, labelled_target_features_val, labelled_target_labels_val, labelled_target_features_test, labelled_target_labels_test = self.get_target_dataset()
            
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
            unlabelled_target_dataset_features, _ = fetch_dataset(target_unlabelled, labelled = False, target = True)
            
            # combine source dataset and unlabelled target dataset into one dataset
            concatenated_train_dataset = CustomConcatDataset(source_dataset, unlabelled_target_dataset_features)

            sampler = BatchSampler(RandomSampler(concatenated_train_dataset), batch_size=batch_size, drop_last=False)
            train_dataloader = DataLoader(dataset=concatenated_train_dataset, sampler = sampler, num_workers=num_workers)            
            #self.train_dataloader = DataLoader(concatenated_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            del concatenated_train_dataset
            del unlabelled_target_dataset_features
            gc.collect()

            # create test dataloader
            combined_val_features = torch.cat(val_features)
            combined_val_labels = torch.cat(val_labels)
            labelled_target_dataset_test = TensorDataset(combined_val_features, combined_val_labels)
            sampler = BatchSampler(RandomSampler(labelled_target_dataset_test), batch_size=self.batch_size, drop_last=False)
            test_dataloader = DataLoader(dataset=labelled_target_dataset_test, sampler = sampler, num_workers=self.num_workers)               
            del labelled_target_dataset_test
            gc.collect()

            return train_dataloader, test_dataloader


        elif dset_type == "semi-supervised":
            # fetch source datasets
            source_features = []
            source_labels = []

            val_features = []
            val_labels = []

            for source_name in self.sources:
                train_features, val_features, train_labels, val_labels = self.fetch_dataset(source_name, labelled = True, target = False, return_val = True)
                source_features.append(train_features)
                source_labels.append(train_labels)

                val_features.append(val_features)
                val_labels.append(val_labels)

                
            # fetch labelled target dataset
            labelled_target_features_train, labelled_target_labels_train, labelled_target_features_val, labelled_target_labels_val, labelled_target_features_test, labelled_target_labels_test = self.get_target_dataset()
            
            # concatenate datasets
            combined_source_features = torch.cat(source_features)
            combined_source_labels = torch.cat(source_labels)
            
            # create source dataloader
            source_dataset = TensorDataset(combined_source_features, combined_source_labels)
            sampler = BatchSampler(RandomSampler(source_dataset), batch_size=self.batch_size, drop_last=True)
            source_dataloader = DataLoader(dataset=source_dataset, sampler = sampler, num_workers=self.num_workers)            
            
            del source_features
            del source_labels
            gc.collect()

            # create labelled target dataloader
            labelled_target_dataset_train = TensorDataset(labelled_target_features_val, labelled_target_labels_val)

            sampler = BatchSampler(RandomSampler(labelled_target_dataset_train), batch_size=min(self.batch_size, len(labelled_target_dataset_train)), drop_last=True)
            labelled_target_dataloader = DataLoader(dataset=labelled_target_dataset_train, sampler = sampler, num_workers=self.num_workers)            

            # fetch unlabelled target dataset and create dataloader
            unlabelled_target_dataset_features, _ = self.fetch_dataset(self.target_unlabelled, labelled = False, target = True)
            unlabelled_target_dataset = TensorDataset(unlabelled_target_dataset_features)
            sampler = BatchSampler(RandomSampler(unlabelled_target_dataset), batch_size=2 * self.batch_size, drop_last=True)
            unlabelled_target_dataloader = DataLoader(dataset=unlabelled_target_dataset, sampler = sampler, num_workers=self.num_workers)            
            
            del unlabelled_target_dataset_features

            # create test dataloader
            combined_val_features = torch.cat(val_features)
            combined_val_labels = torch.cat(val_labels)
            labelled_target_dataset_test = TensorDataset(combined_val_features, combined_val_labels)
            sampler = BatchSampler(RandomSampler(labelled_target_dataset_test), batch_size=self.batch_size, drop_last=False)
            test_dataloader = DataLoader(dataset=labelled_target_dataset_test, sampler = sampler, num_workers=self.num_workers)               
            del labelled_target_dataset_test
            gc.collect()

            return labelled_target_dataloader, unlabelled_target_dataloader, test_dataloader

        elif dset_type == "multi-source":
            # fetch source datasets
            source_datasets = []

            val_features = []
            val_labels = []

            for source_name in self.sources:
                train_features, val_features, train_labels, val_labels = self.fetch_dataset(source_name, labelled = True, target = False, return_val = True)
                source_datasets.append(TensorDataset(train_features, train_labels))

                val_features.append(val_features)
                val_labels.append(val_labels)
            
            labelled_target_features_train, labelled_target_labels_train, labelled_target_features_val, labelled_target_labels_val, labelled_target_features_test, labelled_target_labels_test = self.get_target_dataset()

            # add labelled target train dataset to source domains
            source_datasets.append(TensorDataset(labelled_target_features_val, labelled_target_labels_val))
            
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
            unlabelled_target_dataloader = DataLoader(dataset=unlabelled_target_dataset, sampler = sampler, num_workers=self.num_workers)            
            
            del unlabelled_target_dataset_features

            # create test dataloader
            combined_val_features = torch.cat(val_features)
            combined_val_labels = torch.cat(val_labels)
            labelled_target_dataset_test = TensorDataset(combined_val_features, combined_val_labels)
            sampler = BatchSampler(RandomSampler(labelled_target_dataset_test), batch_size=self.batch_size, drop_last=False)
            test_dataloader = DataLoader(dataset=labelled_target_dataset_test, sampler = sampler, num_workers=self.num_workers)               
            del labelled_target_dataset_test
            gc.collect()

            return source_dataloader_list, unlabelled_target_dataloader, test_dataloader

def create_model(model_type):
        
    from src.model.feature_extractors import BERT_cnn
    feature_extractor = BERT_cnn()
    output_hidden_states = True
    
    if task_classifier.lower() == "dann_task_classifier":
        from src.model.task_classifiers import DANN_task_classifier
        task_classifier = DANN_task_classifier()
    else:
        raise ValueError("Can't find the task classifier name. \
        Please specify the task classifier class name as key in experiment settings of the current experiment.")
        
    if domain_classifier.lower() == "dann_domain_classifier":
        from src.model.domain_classifiers import DANN_domain_classifier
        domain_classifier = DANN_domain_classifier()
    else:
        raise ValueError("Can't find the domain classifier name. \
        Please specify the domain classifier class name as key in experiment settings of the current experiment.")

    if model_type.lower() == "dann":
        model = DANN_model(feature_extractor, task_classifier, domain_classifier, output_hidden_states).to(self.device)  
    elif model_type.lower() == "dirt-t":
        model = DIRT_T_model(feature_extractor, task_classifier, domain_classifier, output_hidden_states).to(self.device)
    elif model_type.lower() == "mme":
        model = mme_model(feature_extractor, task_classifier, domain_classifier, output_hidden_states).to(self.device)      
    elif model_type.lower() == "lirr":
        model = lirr_model(feature_extractor, task_classifier, domain_classifier, output_hidden_states).to(self.device) 
    elif model_type.lower() == "mdan":
        model = mdan_model(feature_extractor, task_classifier, domain_classifier, output_hidden_states).to(self.device) 
    elif model_type.lower() == "m3sda":
        model = m3sda_model(feature_extractor, task_classifier, domain_classifier, output_hidden_states).to(self.device) 
    
    return model

def train_dann(config, checkpoint_dir=None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dset_type = "unsupervised"
    model_type = "dann"
    
    #net = Net(config["l1"], config["l2"])
    model = create_model(model_type)

    model.to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=config["lr"], momentum=config["momentum"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_loader, test_loader = get_data_loaders(dset_type)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(10):  # loop over the dataset multiple times
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(test_loader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be passed as the `checkpoint_dir`
        # parameter in future iterations.
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")

    

def test_accuracy(net, device="cpu"):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


if __name__ == "__main__":
    seed = 123
    # global seed
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        required=True,
        help="The model to perform hyperoptimization for. E.g. 'dann', 'dirt-t', 'mme' etc.",
        )

    # You can change the number of GPUs per trial here:
    #main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)
    args, _ = parser.parse_known_args()

    # for early stopping
    sched = AsyncHyperBandScheduler()

    if args.model == "dann":
        function_to_run = train_dann
        config_dict = {
            "lr": tune.loguniform(1e-4, 1e-2),
            "momentum": tune.uniform(0.1, 0.9),
        }
    elif args.model == "dirt-t":
        function_to_run = train_dirt_t
        config_dict = {
            "lr": tune.loguniform(1e-4, 1e-2),
            "momentum": tune.uniform(0.1, 0.9),
        }
    elif args.model == "mme":
        function_to_run = train_mme
        config_dict = {
            "lr": tune.loguniform(1e-4, 1e-2),
            "momentum": tune.uniform(0.1, 0.9),
        }
    elif args.model == "lirr":
        function_to_run = train_lirr
        config_dict = {
            "lr": tune.loguniform(1e-4, 1e-2),
            "momentum": tune.uniform(0.1, 0.9),
        }
    elif args.model == "mdan":
        function_to_run = train_mdan
        config_dict = {
            "lr": tune.loguniform(1e-4, 1e-2),
            "momentum": tune.uniform(0.1, 0.9),
        }
    elif args.model == "m3sda":
        function_to_run = train_m3sda
        config_dict = {
            "lr": tune.loguniform(1e-4, 1e-2),
            "momentum": tune.uniform(0.1, 0.9),
        }

    result = tune.run(
        function_to_run,
        metric="mean_accuracy",
        mode="max",
        name="exp",
        scheduler=sched,
        stop={
            "mean_accuracy": 0.98,
            "training_iteration": 5 if args.smoke_test else 100
        },
        resources_per_trial={
            "cpu": 2,
            "gpu": int(args.cuda)  # set this for GPUs
        },
        num_samples=1 if args.smoke_test else 50,
        config=config_dict)

    print("Best config is:", result.best_config)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
