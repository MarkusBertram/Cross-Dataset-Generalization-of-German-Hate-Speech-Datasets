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
from ray.air.config import ScalingConfig
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
from src.model.DIRT_T_model import DIRT_T_model
import sys
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB
from transformers import BertModel
import copy
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.tuner import Tuner, TuneConfig
from ray.air.config import RunConfig
from ray.train.torch import TorchCheckpoint, TorchTrainer
from ray import train
from ray.tune.search.hyperopt import HyperOptSearch
class VATLoss(nn.Module):

    """ Virtual Adversarial Training Loss function
    Reference:
    TODO
    """

    def __init__(self, radius = 1): #model, radius=1):

        super(VATLoss, self).__init__()
        #self.model  = model
        self.radius = 1

        self.loss_func_nll = KLDivWithLogits()

    def forward(self, bert_output, p, model):
        # x: input features
        # p: task classifier softmax probabilities of input features x
        stacked_output = torch.stack(bert_output[2])
        x = torch.swapaxes(stacked_output, 0, 1)
        # get random vector of size x
        eps = torch.randn_like(x)
        # normalize random vector and multiply by e-6
        eps = 1e-6*F.normalize(eps)

        eps.requires_grad = True
        # calculate output of x + random vector
        eps_p = model(x + eps, input_is_bert=False)[0]
        # calculate KL divergence of output of x + random vector and output of x
        loss  = self.loss_func_nll(eps_p, p.detach())

        # calculate gradient of KL divergence
        grad = torch.autograd.grad(loss, [eps], retain_graph=True)[0]
        eps_adv = grad.detach()

        # normalize gradient
        eps_adv = F.normalize(eps_adv)#normalize_perturbation(eps_adv)
        # adversarial x is x + 1 * gradient
        x_adv = x + self.radius * eps_adv
        x_adv = x_adv.detach()

        p_adv, _ = model.forward(x_adv, input_is_bert=False)
        loss     = self.loss_func_nll(p_adv, p.detach())

        return loss

class ConditionalEntropy(nn.Module):

    """ estimates the conditional cross entropy of the input
    $$
    \frac{1}{n} \sum_i \sum_c p(y_i = c | x_i) \log p(y_i = c | x_i)
    $$
    By default, will assume that samples are across the first and class probabilities
    across the second dimension.
    """

    def forward(self, input):
        p     = torch.sigmoid(input)
        log_p = F.logsigmoid(input)

        H = - torch.mean(p * log_p)

        return H

class KLDivWithLogits(nn.Module):

    def __init__(self):

        super(KLDivWithLogits, self).__init__()

        self.kl = nn.KLDivLoss(size_average=False, reduce=True)


    def forward(self, x, y):

        log_p = F.logsigmoid(x)
        q     = torch.sigmoid(y)

        return self.kl(log_p, q) / x.size()[0]

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
        
        unlabelled_size += len(labelled_target_features_train)

        source_features.append(labelled_target_features_train)
        source_labels.append(labelled_target_labels_train)
        val_features_list.append(labelled_target_features_val)
        val_labels_list.append(labelled_target_labels_val)

        combined_source_features = torch.cat(source_features)
        combined_source_labels = torch.cat(source_labels)
        
        # concatenate datasets
        source_dataset = TensorDataset(combined_source_features, combined_source_labels)

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

    model = DIRT_T_model(feature_extractor, task_classifier, domain_classifier, output_hidden_states).to(device)  
    
    return model
      
def train_dirt_t(config):
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

    lr =  1.4e-5
    beta1 = 0.913
    beta2 = 0.993

    truncation_length= 512
    
    model = create_model(device, truncation_length)
    model = train.torch.prepare_model(model)
    model.to(device)
    
    train_loader, test_loader = get_data_loaders(
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
    
    optimizer = optim.Adam(
        model.parameters(), lr=lr, betas=(beta1, beta2))
    step = 0

    # To restore a checkpoint, use `session.get_checkpoint()`.
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state, step = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    for name, param in model.named_parameters():
        if "bert" in name:
            param.requires_grad = False

    bert = BertModel.from_pretrained("deepset/gbert-base").to(device)
    output_hidden_states = True
    disc = nn.BCEWithLogitsLoss()
    src_vat     = VATLoss()
    tgt_vat     = VATLoss()
    conditionE  = ConditionalEntropy()
    crossE      = nn.BCEWithLogitsLoss()

    lambda_d = config["lambda_d"]
    lambda_s = config["lambda_s"]
    lambda_t = config["lambda_t"]
    beta = config["beta"]
    polyak_factor = config["polyak_factor"]

    epochs = 10
    while True:  # loop over the dataset multiple times
        model.train()
        if not loaded_checkpoint:
            for epoch in range(1, epochs):
                for i, (source_batch, unlabelled_target_features) in enumerate(train_loader):
                                    
                    #source_features = source_batch[0][0].to(device)
                    source_labels = source_batch[1][0].to(device)

                    source_input_ids = source_batch[0][0][:,0].to(device)
                    source_attention_mask = source_batch[0][0][:,1].to(device)

                    unlabelled_target_features = unlabelled_target_features[0].to(device)

                    source_bert_output = bert(input_ids=source_input_ids, attention_mask=source_attention_mask, return_dict = False, output_hidden_states=output_hidden_states)
                    target_bert_output = bert(input_ids=unlabelled_target_features[:,0], attention_mask=unlabelled_target_features[:,1], return_dict = False, output_hidden_states=output_hidden_states)
                    
                    source_class_output, source_domain_output = model(input_features=source_bert_output)
                    target_class_output, target_domain_output = model(input_features=target_bert_output)

                    # Cross-Entropy Loss of Source Domain, Source Generalization Error
                    crossE_loss = crossE(source_class_output, source_labels)

                    # conditional entropy with respect to target distribution, enforces cluster assumption
                    conditionE_loss = conditionE(target_class_output)               

                    # Domain Discriminator, Divergence of Source and Target Domain
                    domain_loss = .5*(
                    disc(source_domain_output,torch.zeros_like(source_domain_output)) + 
                    disc(target_domain_output, torch.ones_like(target_domain_output))
                    )
                    
                    vat_src_loss = src_vat(source_bert_output, source_class_output, model)
                    vat_tgt_loss = tgt_vat(target_bert_output, target_class_output, model)

                    disc_loss = 0.5 *(
                    disc(source_domain_output,torch.ones_like(source_domain_output)) + 
                    disc(target_domain_output, torch.zeros_like(target_domain_output))
                    )

                    loss = crossE_loss + lambda_d*domain_loss + lambda_d*disc_loss +  lambda_s*vat_src_loss + lambda_t*vat_tgt_loss + lambda_t*conditionE_loss
                    optimizer.zero_grad(set_to_none=True)

                    loss.backward()

                    optimizer.step()   

        ############ DIRT_T Training

        teacher = copy.deepcopy(model)
    
        for param in teacher.parameters():
            param.requires_grad = False

        #optimizer   = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
        #optimizer2  = WeightEMA(teacher_params, student_params)#DelayedWeight(teacher_params, student_params)
        
        crossE      = nn.BCEWithLogitsLoss().to(device)
        conditionE  = ConditionalEntropy().to(device)
        tgt_vat     = VATLoss().to(device)#VATLoss(model, radius=radius).to(device)
        dirt        = KLDivWithLogits() #F.kl_div().to(device)#KLDivWithLogits()    

  #     teacher.eval()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "bert" in name:
                    param.requires_grad = False
        with torch.no_grad():
            for name, param in teacher.named_parameters():
                if "bert" in name:
                    param.requires_grad = False

        while True:
            model.train()

            for i, (source_batch, unlabelled_target_features) in enumerate(train_loader):

                optimizer.zero_grad(set_to_none=True)
                #optimizer2.zero_grad()

                unlabelled_target_features = unlabelled_target_features[0].to(device)
                target_bert_output = bert(input_ids=unlabelled_target_features[:,0], attention_mask=unlabelled_target_features[:,1], return_dict = False, output_hidden_states=output_hidden_states)

                if output_hidden_states == False:
                    target_bert_output = target_bert_output[0][:,0,:]

                target_class_output, target_domain_output = model(input_features=target_bert_output)
                teacher_target_class_output, teacher_target_domain_output = teacher(input_features=target_bert_output)

                conditionE_loss = conditionE(target_class_output) # conditional entropy
                dirt_loss       = dirt(target_class_output, teacher_target_class_output)

                vat_tgt_loss    = tgt_vat(target_bert_output, target_class_output, model)

                loss = lambda_t*conditionE_loss + lambda_t*vat_tgt_loss + beta*dirt_loss 
                
                loss.backward()
                optimizer.step()
                #optimizer2.step()

            # polyak averaging
            # https://discuss.pytorch.org/t/copying-weights-from-one-net-to-another/1492/17
            for target_param, param in zip(teacher.parameters(), model.parameters()):
                target_param.data.copy_(polyak_factor*param.data + target_param.data*(1.0 - polyak_factor))


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

            step += 1

            os.makedirs("my_model", exist_ok=True)
            torch.save(
                (model.state_dict(), optimizer.state_dict(), step), "my_model/checkpoint.pt")
            checkpoint = Checkpoint.from_directory("my_model")
            session.report({"loss": avg_val_loss}, checkpoint=checkpoint)
        
    print("Finished Training")

if __name__ == "__main__":
    config_dict = {
      "train_loop_config": {
            "lambda_d": tune.loguniform(1e-3, 1e-1),
            "lambda_s": tune.uniform(0, 1),
            "lambda_t": tune.loguniform(1e-3, 1e-1),
            "beta": tune.loguniform(1e-5, 1e-1),
            "polyak_factor": tune.loguniform(0.99, 0.9999)
      }
    }

    trainer = TorchTrainer(
        train_loop_per_worker=train_dirt_t,
        train_loop_config=config_dict,
        scaling_config=ScalingConfig(
            num_workers=1,  # Number of workers to use for data parallelism.
            use_gpu = True
        )
    )

    asha_scheduler = ASHAScheduler(
        max_t=10,
    )

    search_alg = HyperOptSearch()
    tune_config = TuneConfig(
        metric="loss",
        mode="min",
        num_samples=-1,
        search_alg=search_alg,
        scheduler= asha_scheduler
    )

    tuner = Tuner(
        trainer,
        param_space=config_dict,
        tune_config=tune_config
    )

    # Execute tuning.
    result_grid = tuner.fit()

    # Fetch the best result.
    best_result = result_grid.get_best_result()
    print("Best Result:", best_result)

    print("Best trial config: {}".format(best_result.config))
