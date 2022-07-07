#from feature_extractors import BERT_cnn, bert_cls
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, BatchEncoding, Trainer, TrainingArguments, AdamW

import torch
from torch import nn
from transformers import BertModel
from torch.nn import CrossEntropyLoss
import gc
import sys

def get_model(
    config,
    **kwargs,
):
    """get_model [[function which returns instance of the experiments model]]
    [extended_summary]
    Args:
        feature_extractor ([string]): ["bert_cls":bert with cls token, "bert_cnn": bert with cnn].
        task_classifier ([string]): ["fd1": X hidden layers]

        similarity ([type], optional): [For genOdinMode "E":Euclidean distance, "I": FC Layer, "C": Cosine Similarity, "IR": I-reversed , "ER": E-revesered, "CR": "C-reversed" ]. Defaults to None.
        num_classes (int, optional): [Number of classes]. Defaults to 10.
        include_bn (bool, optional): [Include batchnorm]. Defaults to False.
        channel_input (int, optional): [dataset channel]. Defaults to 3.
    Raises:
        ValueError: [When model name false]
    Returns:
        [nn.Module]: [parametrized Neural Network]
    """

    if config["feature_extractor"] == "BERT_cls":
        from feature_extractors import BERT_cls
        feature_extractor = BERT_cls()
        output_hidden_states = False
    elif config["feature_extractor"] == "BERT_cnn":
        from feature_extractors import BERT_cnn
        feature_extractor = BERT_cnn()
        output_hidden_states = True
    else:
        print("error, can't find this feature extractor. please specify bert_cls or bert_cnn in experiment settings.")
    
    if config["task_classifier"] == "tc1":
        from task_classifiers import task_classifier1
        task_classifier = task_classifier1()
        
    if config["domain_classifier"] == "dc1":
        from domain_classifiers import domain_classifier1
        domain_classifier = domain_classifier1()

    if config["model"] == "DANN":
        from model.DANN_model import DANN_model     
        return DANN_model(feature_extractor, task_classifier, domain_classifier, output_hidden_states)


config = {"model":"DANN","feature_extractor": "BERT_cnn", "task_classifier": "tc1", "domain_classifier": "dc1" }
model = get_model(config)
print(model)