#from feature_extractors import BERT_cnn, bert_cls
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, BatchEncoding, Trainer, TrainingArguments, AdamW
from functions import ReverseLayerF
import torch
from torch import nn
from transformers import BertModel
from torch.nn import CrossEntropyLoss
import gc
import sys

def get_dirt_t_model(feature_extractor_module, task_classifier_module, domain_classifier_module, output_hidden_states):

    class DIRT_T_model(nn.Module):
            def __init__(self):
                super(DIRT_T_model, self).__init__()
                self.bert = BertModel.from_pretrained("deepset/gbert-base")
                self.output_hidden_states = output_hidden_states
                self.feature_extractor = feature_extractor_module
                self.task_classifier = task_classifier_module
                self.domain_classifier = domain_classifier_module

            def forward(
            self,
            input,
            alpha
            ):
                bert_output = self.bert(input["input_ids"], attention_mask=input["attention_mask"], output_hidden_states=self.output_hidden_states)
                feature_extractor_output = self.feature_extractor(bert_output)

                class_output = self.task_classifier(feature_extractor_output)
                
                domain_output = self.domain_classifier(feature_extractor_output)

                return class_output, domain_output

    return DIRT_T_model