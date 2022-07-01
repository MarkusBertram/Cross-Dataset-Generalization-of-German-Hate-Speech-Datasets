import numpy as np
import pandas as pd
from datetime import datetime
import datasets
import os
import scipy as sp
import gc
import time
import torch
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import re
import unicodedata
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, BatchEncoding, Trainer, TrainingArguments, AdamW
from utils.utils import fetch_import_module
from pipelines import utils_pipeline
from time import gmtime, strftime
from tqdm import tqdm
from datasets import concatenate_datasets,Dataset,ClassLabel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sys
from torch.utils.data import Dataset, TensorDataset, DataLoader
import subprocess
from accelerate import Accelerator
os.environ['MKL_THREADING_LAYER'] = 'GNU'
#from sklearn.model_selection import train_test_split
from pathlib import Path

def cleanTweets(dataset):
    twitter_username_re = re.compile(r'@([A-Za-z0-9_]+)')
    hashtag_re = re.compile(r'\B(\#[a-zA-Z0-9]+\b)(?!;)')
    html_symbol_re = re.compile(r'&[^ ]+')
    url_re = re.compile(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})')
    for tweet in dataset:
        text = twitter_username_re.sub("[UNK]",tweet['text'])
        text = unicodedata.normalize('NFKC',text)
        text = text.replace('\n',' ')
        text = text.replace('RT ',' ')
        text = hashtag_re.sub("[UNK]",text)
        text = html_symbol_re.sub(" ",text)
        text = url_re.sub("[UNK]",text)
        tweet['text'] = text
    return dataset

def getPaths(root_path): 
    path_models = Path(root_path+'models/')
    path_datasets =Path(root_path+'datasets/')
    path_output = Path(root_path+'outputs/')
    path_logs = Path(root_path+'logs/')
    path_models.mkdir(parents=True, exist_ok=True)
    path_datasets.mkdir(parents=True, exist_ok=True)
    path_output.mkdir(parents=True, exist_ok=True)
    path_logs.mkdir(parents=True, exist_ok=True)
    return path_models,path_datasets,path_output,path_logs

def prepareData(data):
    data = cleanTweets(data)
    return convertLabelsToInt(utils_pipeline.get_huggingface_dataset_format(data))

def convertLabelsToInt(dataset):
    label_to_int = {
        "neutral": 0,
        "abusive": 1
    }
    #dataset['label'] = dataset["label"].map(label_to_int)
    dataset = dataset.map(lambda convertLabels: {"label": label_to_int[convertLabels["label"]]})
    return dataset

# define a prediction function
def f(x, model, tokenizer):
    tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=512, truncation=True) for v in x]).cuda()
    outputs = model(tv)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores[:,1]) # use one vs rest logit units
    return val

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def transform_to_dataset(dataset, tokenizer):
    input_ids = []
    attention_masks = []
    targets = []

    # get sentence vector
    for index, row in dataset.iterrows():
        #text = preprocessing_multilingual.clean_text(sentence['text'])
        # tokenize text
        encoded_dict = tokenizer.encode_plus(
            row["text"],
            add_special_tokens = True,
            padding = 'max_length',
            max_length = tokenizer.model_max_length,
            truncation = True,
            return_attention_mask = True,
            return_tensors = 'pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        
        #label = dset_name + "_" + str(sentence['label'])
        targets.append(row["label"])

    targets = torch.as_tensor(targets)

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    stacked_input = torch.stack((input_ids, attention_masks), dim=1)
    
    tensordataset = TensorDataset(stacked_input, targets)

    return tensordataset

def plotMatrix(number_of_runs, eval_metrics,labels,results_dir, fair, selected_type='f1', type_name=""):

    path_fig = results_dir
    sns.set(font_scale=1.0)

    if fair == True:
        average_matrix = []
        avg_of_avg_classifiers = []
        for eval_metric in eval_metrics:
            matrix = np.empty([len(eval_metric),len(eval_metric)])
            for i in range(len(eval_metric)):
                for j in range(len(eval_metric[i][selected_type])-1):
                    matrix[i][j] = eval_metric[i][selected_type][j]
            
            average_matrix.append(matrix)
            # calculate averages
            avg_classifiers = []
            avg_testsets = []
            size = len(matrix[0])

            for i in range(len(eval_metric)):
                avg_classifiers.append(eval_metric[i][selected_type][-1])
            vec = np.asarray(avg_classifiers).reshape(size,1)
            avg_of_avg_classifiers.append(vec)

        # averaging of the runs:
        matrix = np.mean(average_matrix, axis = 0)
        avg_classifiers = np.mean(avg_of_avg_classifiers, axis = 0)
        
        min_val = np.amin(matrix)
        max_val = np.amax(matrix) 

        fig = plt.figure(figsize=(11,13))
        ax1 = plt.subplot2grid((10,9), (0,0), colspan=6, rowspan=7)
        ax3 = plt.subplot2grid((10,9), (0,8), rowspan=7)

        cmap = "Blues"
        center = matrix[0][0]
        sns.set(font_scale=0.8)
        hm1 = sns.heatmap(matrix, ax=ax1,annot=True, fmt=".1%",vmin=min_val, vmax=max_val, cbar=False,cmap=cmap,square=True,xticklabels=labels, yticklabels=labels)
        hm2 = sns.heatmap(avg_classifiers, ax=ax3, annot=True, fmt=".1%", cbar=False, xticklabels=False, yticklabels=False,vmin=min_val, vmax=max_val,cmap=cmap,square=True)
        hm1.set_xticklabels(labels, rotation=90, ha='center')
        
        
        ax1.set_title(type_name)
        ax1.xaxis.tick_top()
        ax1.tick_params(length=0)
        ax1.set(xlabel='Test sets', ylabel='Classifiers')
        ax1.xaxis.set_label_coords(0.5, 1.4)

        ax3.set(xlabel='Combined\n test set', ylabel='')
        #ax3.xaxis.tick_top()
        ax3.xaxis.set_label_coords(0.5, 1.13)
        
        fig.savefig(path_fig + "classification_cross_" + selected_type +".pdf", bbox_inches='tight', dpi=300)
        fig.savefig(path_fig + "classification_cross_" + selected_type +".png", bbox_inches='tight', dpi=300)
    else:
        average_matrix = []#np.empty([len(eval_metrics[0]),len(eval_metrics[0])])
        avg_of_avg_classifiers = []
        for eval_metric in eval_metrics:
            matrix = np.empty([len(eval_metric),len(eval_metric)])
            for i in range(len(eval_metric)):
                for j in range(len(eval_metric[i][selected_type])):
                    matrix[i][j] = eval_metric[i][selected_type][j]
            
            average_matrix.append(matrix)
            # calculate averages
            avg_classifiers = []
            avg_testsets = []
            size = len(matrix[0])


        # averaging of the runs:
        matrix = np.mean(average_matrix, axis = 0)
        
        min_val = np.amin(matrix)
        max_val = np.amax(matrix) 

        fig = plt.figure(figsize=(11,13))
        ax1 = plt.subplot2grid((10,9), (0,0), colspan=6, rowspan=7)

        cmap = "Blues"
        center = matrix[0][0]
        sns.set(font_scale=0.8)
        hm1 = sns.heatmap(matrix, ax=ax1,annot=True, fmt=".1%",vmin=min_val, vmax=max_val, cbar=False,cmap=cmap,square=True,xticklabels=labels, yticklabels=labels)
        hm1.set_xticklabels(labels, rotation=90, ha='center')
        
        
        ax1.set_title(type_name)
        ax1.xaxis.tick_top()
        ax1.tick_params(length=0)
        ax1.set(xlabel='Test sets', ylabel='Classifiers')
        ax1.xaxis.set_label_coords(0.5, 1.4)
        
        fig.savefig(path_fig + "classification_cross_" + selected_type +".pdf", bbox_inches='tight', dpi=300)
        fig.savefig(path_fig + "classification_cross_" + selected_type +".png", bbox_inches='tight', dpi=300)
# tokenize datasets
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=512)

if __name__ == '__main__':
    config = yaml.safe_load(open("settings/config.yaml"))
    dataset_names = list(config['datasets'].keys())
    data_sets_text = []
    for dset in dataset_names:
        dset_module = fetch_import_module(dset)
        data_sets_text.append(dset_module.get_data_binary())

    number_of_runs = 10
    fair = False
    SPLIT_RATIO = 0.2
    COMBINED_RATIO = 0.5
    model_name= 'deepset/gbert-base'
    path = './tmp3/'
    number_of_tokens = 50
    batch = 16
    num_epochs = 3
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = accelerator.device
    # get paths and create folders
    path_models,path_datasets,path_output,path_logs =getPaths(path)

    print('-'*50)
    print('Loading data sets...')
    print('-'*50)
    
    # prepare data sets
    data_sets = []
    for dataset in data_sets_text:
        dset = prepareData(dataset)
        dset = dset.class_encode_column("label")
        data_sets.append(dset)

    del data_sets_text
    del dset
    print('-'*50)
    print('Preparing data sets...')
    print('-'*50)

    if fair == True:
        # find lengths of smallest data set
        min_length = 99999999
        for dataset in data_sets:
            min_length = min(min_length,len(dataset))
        size_train = round(min_length*(1-SPLIT_RATIO))
        size_test = min_length - size_train
    else:
        size_train = 1-SPLIT_RATIO
        size_test = SPLIT_RATIO

    for run_iter in range(number_of_runs):
        # split data into train and test set  
        training_sets = []
        test_sets = []
        combined_test_set = None
        for i,dataset in enumerate(data_sets):
            
            ds_dict = {}
            ds_dict_1 = {}
            ds_dict_2 = {}
            
            ds_dict = dataset.train_test_split(test_size=size_test,train_size=size_train, stratify_by_column = "label", shuffle=True)
            
            training_sets.append(ds_dict['train'])
            test_sets.append(ds_dict['test'])

            # combined test set
            if fair == True:
                ds_dict_2 = ds_dict['test'].train_test_split(train_size=COMBINED_RATIO,stratify_by_column = "label",shuffle=True)
            
                if combined_test_set is None:
                    combined_test_set = ds_dict_2['train']
                else:
                    #combined_test_set = pd.concat([combined_test_set,ds_dict_2['train']])
                    combined_test_set = concatenate_datasets([combined_test_set,ds_dict_2['train']])
        if combined_test_set is not None:
            test_sets.append(combined_test_set)
            path_combined_test = path_datasets / 'combined_test'
            Path(path_combined_test).mkdir(parents=True, exist_ok=True)
            combined_test_set.save_to_disk(path_combined_test)            

        # train and evaluate classifiers
        for i in tqdm(range(len(data_sets))):
            path_model = path_models / "{}_{}_model".format(str(i),dataset_names[i])

            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

            train_dataset = training_sets[i].map(tokenize, batched=True, batch_size=len(training_sets[i]))
            train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
            
            # define trainer
            training_args = TrainingArguments(
            output_dir=path_model,          # output directory
            num_train_epochs=num_epochs,              # total # of training epochs
            per_device_train_batch_size=batch,  # batch size per device during training
            per_device_eval_batch_size=64,   # batch size for evaluation
            learning_rate = 2e-5
        )

            trainer = Trainer(
                model=model,                         # the instantiated 🤗 Transformers model to be trained
                args=training_args,                  # training arguments, defined above
                train_dataset=train_dataset,         # training dataset
                compute_metrics=compute_metrics,
            )
            
            # train model
            trainer.train()  
            
            # evaluate model
            accuracy = []
            f1 = []
            precision = []
            recall = []

            for j in range(len(test_sets)):
                # prepare evaluation test set
                eval_dataset = test_sets[j].map(tokenize, batched=True, batch_size=len(test_sets[j]))
                eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
                
                # predict
                results_it = trainer.predict(eval_dataset) 
                accuracy.append(results_it.metrics["test_accuracy"])
                f1.append(results_it.metrics['test_f1'])
                precision.append(results_it.metrics['test_precision'])
                recall.append(results_it.metrics['test_recall'])

                
            results = {}
            results['f1'] = f1
            results['precision'] = precision
            results['recall'] = recall
            results['accuracy'] = accuracy
            #results['predictions'] = predictions
                
            # # save model
            # trainer.save_model(path_model)

            pickle.dump(results, open("{}{}{}_{}.pkl".format(path_output,str(run_iter),str(i),dataset_names[i]), "wb"))

            # clear GPU memory
            del model, trainer
            gc.collect()

    evaluation_results = []
    for r in range(number_of_runs):
        multiple_results = []
        for i in range(len(data_sets)):
            file = "{}{}{}_{}.pkl".format(path_output,str(r),str(i),dataset_names[i])
            single_result = pickle.load(open(file, "rb"))
            multiple_results.append(single_result)
        evaluation_results.append(multiple_results)
    now = datetime.now()
    results_dir = "./results/"+"avg-cross-dataset_performance_"+now.strftime("%Y%m%d-%H%M%S")+"/"
    if os.path.exists(results_dir) == False:
        os.makedirs(results_dir)
    print("Macro F1")
    plotMatrix(number_of_runs, evaluation_results,dataset_names, results_dir, fair, selected_type="f1")
    print("Precision")
    plotMatrix(number_of_runs, evaluation_results,dataset_names, results_dir, fair, selected_type="precision")
    print("Recall")
    plotMatrix(number_of_runs, evaluation_results,dataset_names, results_dir, fair, selected_type="recall")
    print("Accuracy")
    plotMatrix(number_of_runs, evaluation_results,dataset_names, results_dir, fair, selected_type="accuracy")

    
