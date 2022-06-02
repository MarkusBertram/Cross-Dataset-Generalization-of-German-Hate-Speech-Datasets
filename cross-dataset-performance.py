import numpy as np
import pandas as pd
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, BatchEncoding, Trainer, TrainingArguments, AdamW, get_scheduler
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
from sklearn.model_selection import train_test_split
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

# def tokenize(df, tokenizer):

#     for sentence in df["text"]:
        
#         BatchEncoding = tokenizer.encode_plus(
#                 sentence,
#                 add_special_tokens = True,
#                 padding = 'max_length',
#                 max_length = tokenizer.model_max_length,
#                 truncation = True,
#                 return_attention_mask = True,
#                 return_tensors = 'pt'
#             )

#     tokenized_df = pd.DataFrame(data = {"input_ids" : BatchEncoding["input_ids"], "token_type_ids" : BatchEncoding["token_type_ids"], "attention_mask" : BatchEncoding["attention_mask"], "label": df["label"]})

#     return tokenized_df
# # tokenize datasets
# def tokenize(batch, tokenizer=tokenizer):
#     return tokenizer(batch, padding=True, truncation=True, max_length=512)

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

    # le = preprocessing.LabelEncoder()
    # targets = le.fit_transform(labels)
    targets = torch.as_tensor(targets)

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    stacked_input = torch.stack((input_ids, attention_masks), dim=1)
    
    tensordataset = TensorDataset(stacked_input, targets)

    return tensordataset

def plotMatrix(eval_metrics,labels,selected_type='f1', type_name=""):
    path_fig = "./results/"+strftime("%Y%m%d", gmtime())+ "-" + "-".join(labels).replace(" ","_")

    sns.set(font_scale=1.0)
    matrix = np.empty([len(eval_metrics),len(eval_metrics)])
    for i in range(len(eval_metrics)):
        for j in range(len(eval_metrics[i][selected_type])-1):
            matrix[i][j] = eval_metrics[i][selected_type][j]

    # calculate averages
    avg_classifiers = []
    avg_testsets = []

    for i in range(len(eval_metrics)):
        avg_classifiers.append(eval_metrics[i][selected_type][-1])   


    size = len(matrix[0])
    min_val = np.amin(matrix)
    max_val = np.amax(matrix) 

    avg_classifiers = np.asarray(avg_classifiers).reshape(size,1)

    fig = plt.figure(figsize=(6,5))
    ax1 = plt.subplot2grid((6,5), (0,0), colspan=4, rowspan=5)
    ax3 = plt.subplot2grid((6,5), (0,4), rowspan=5)

    cmap = "Blues"
    center = matrix[0][0]

    hm1 = sns.heatmap(matrix, ax=ax1,annot=True, fmt=".1%",vmin=min_val, vmax=max_val, cbar=False,cmap=cmap,square=True,xticklabels=labels, yticklabels=labels)
    hm2 = sns.heatmap(avg_classifiers, ax=ax3, annot=True, fmt=".1%", cbar=False, xticklabels=False, yticklabels=False,vmin=min_val, vmax=max_val,cmap=cmap,square=True)
    hm1.set_xticklabels(labels, rotation=90, ha='center')
    
    
    ax1.set_title(type_name)
    ax1.xaxis.tick_top()
    ax1.tick_params(length=0)
    ax1.set(xlabel='Test sets', ylabel='Classifiers')
    ax1.xaxis.set_label_coords(0.5, 1.30)

    ax3.set(xlabel='Combined\n test set', ylabel='')
    #ax3.xaxis.tick_top()
    ax3.xaxis.set_label_coords(0.5, 1.13)
    
    fig.savefig(path_fig + "-classification_cross_" + selected_type +".pdf", bbox_inches='tight', dpi=300)
    fig.savefig(path_fig + "-classification_cross_" + selected_type +".png", bbox_inches='tight', dpi=300)

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
    
    

    SEED =321
    SPLIT_RATIO = 0.15
    COMBINED_RATIO = 0.5
    model_name= 'deepset/gbert-base'
    path = './tmp2/'
    number_of_tokens = 50
    batch = 10
    num_epochs = 1
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
        data_sets.append(prepareData(dataset))

    print('-'*50)
    print('Preparing data sets...')
    print('-'*50)
    
    # find lengths of smallest data set
    min_length = 99999999
    for dataset in data_sets:
        min_length = min(min_length,len(dataset))
    size_train = round(min_length*(1-SPLIT_RATIO))
    size_test = min_length - size_train

    # split data into train and test set  
    training_sets = []
    validation_sets = []
    test_sets = []
    combined_test_set = None
    for i,dataset in enumerate(data_sets):
        ds_dict = {}
        ds_dict_1 = {}
        ds_dict_2 = {}
        # split data sets and tokenize
        ## train/test split
        #tokenized_dataset = tokenize(dataset, tokenizer)
        #ds_dict['train'], ds_dict['test'] = train_test_split(dataset, test_size=size_test,train_size=size_train,shuffle=True)
        ds_dict = dataset.train_test_split(test_size=size_test,train_size=size_train,shuffle=True,seed=SEED)
        
        training_sets.append(ds_dict['train'])
        #validation_sets.append(ds_dict_2['test'])
        test_sets.append(ds_dict['test'])

        # combined test set
        #ds_dict_2['train'], ds_dict_2['test'] = train_test_split(ds_dict['test'],train_size=COMBINED_RATIO,shuffle=True)
        ds_dict_2 = ds_dict['test'].train_test_split(train_size=COMBINED_RATIO,shuffle=True,seed=SEED)
        if combined_test_set is None:
            combined_test_set = ds_dict_2['train']
        else:
            #combined_test_set = pd.concat([combined_test_set,ds_dict_2['train']])
            combined_test_set = concatenate_datasets([combined_test_set,ds_dict_2['train']])
    test_sets.append(combined_test_set)


    # train and evaluate classifiers
    for i in tqdm(range(len(data_sets))):
        path_model = "{}{}_{}_model".format(path_models,str(i),dataset_names[i])
        #train_dataset = transform_to_dataset(training_sets[i], tokenizer)
        # get train, test dataloader
        #train_dataloader = DataLoader(train_dataset, batch_size = 5, shuffle = False)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        #optimizer = AdamW(model.parameters(), lr=3e-5)
        #model, train_dataloader, optimizer = accelerator.prepare(model, train_dataloader, optimizer)
        train_dataset = training_sets[i].map(tokenize, batched=True, batch_size=len(training_sets[i]))
        train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        # define trainer
        training_args = TrainingArguments(
            output_dir=path_model,          # output directory
            num_train_epochs=num_epochs,              # total # of training epochs
            per_device_train_batch_size=batch,  # batch size per device during training
            per_device_eval_batch_size=64,   # batch size for evaluation
            #warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir=path_logs
        )

        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            compute_metrics=compute_metrics,
            #tokenizer = tokenizer
        )
        
        # train model
        trainer.train()
        # num_training_steps = num_epochs * len(train_dataloader)
        # lr_scheduler = get_scheduler(
        #         "linear",
        #         optimizer=optimizer,
        #         num_warmup_steps=0,
        #         num_training_steps=num_training_steps
        #     )
        # model.train()
        # for epoch in range(num_epochs):
        #     for x, y in train_dataloader:
        #         #input_id = x[:, 0].to(device)
        #         #attention_masks = x[:, 1].to(device)
        #         #outputs = model(**batch)
        #         #outputs = model(input_ids = input_id, attention_mask = attention_masks, labels = y)
        #         outputs = model(input_ids = x[:, 0], attention_mask = x[:, 1], labels = y)
        #         loss = outputs.loss
                
        #         accelerator.backward(loss)
        #         optimizer.step()
        #         lr_scheduler.step()
        #         optimizer.zero_grad()        
        
        # evaluate model
        accuracy = []
        f1 = []
        precision = []
        recall = []
        #predictions = []
            
        for j in range(len(test_sets)):
            # prepare evluation test set
            # eval_dataset = test_sets[j].map(tokenize, batched=True, batch_size=len(train_dataset))
            # eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
            eval_dataset = test_sets[j].map(tokenize, batched=True, batch_size=len(train_dataset))
            eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
            
            # predict
            results_it = trainer.predict(eval_dataset) 
            print(results_it)
            accuracy.append(results_it.metrics["test_accuracy"])
            f1.append(results_it.metrics['test_f1'])
            precision.append(results_it.metrics['test_precision'])
            recall.append(results_it.metrics['test_recall'])
            #predictions.append(results)

            
        results = {}
        results['f1'] = f1
        results['precision'] = precision
        results['recall'] = recall
        results['accuracy'] = accuracy
        #results['predictions'] = predictions
            
        # save model
        trainer.save_model(path_model)

        # with path_output.open('wb') as fp:
        #     pickle.dump(results, fp)
        pickle.dump(results, open("{}{}_{}.pkl".format(path_output,str(i),dataset_names[i]), "wb"))

        # clear GPU memory
        del model, trainer
        gc.collect()

    evaluation_results = []
    for i in range(len(data_sets)):
        file = "{}{}_{}.pkl".format(path_output,str(i),dataset_names[i])
        single_result = pickle.load(open(file, "rb"))
        evaluation_results.append(single_result)
    
    print("Macro F1")
    plotMatrix(evaluation_results,dataset_names,selected_type="f1")
    print("Precision")
    plotMatrix(evaluation_results,dataset_names,selected_type="precision")
    print("Recall")
    plotMatrix(evaluation_results,dataset_names,selected_type="recall")
    print("Accuracy")
    plotMatrix(evaluation_results,dataset_names,selected_type="accuracy")

    
