#from feature_extractors import BERT_cnn, bert_cnn
from pyrsistent import s
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, BatchEncoding, Trainer, TrainingArguments, AdamW
#from functions import ReverseLayerF
import torch
from torch import nn
from transformers import BertModel
from torch.nn import CrossEntropyLoss
import gc
import sys
from torch.nn.parameter import Parameter
from torch.autograd import Function

class Noise_layer(nn.Module):
    def __init__(self,mid,w):
        super(Noise_layer, self).__init__()
        self.mid = mid
        self.w = w
        self.noise = Parameter(torch.zeros(mid, w, w).normal_(0, 1))

    def forward(self, input):
        return input + self.noise

class DIRT_T_model(nn.Module):
    def __init__(self, feature_extractor_module, task_classifier_module, domain_classifier_module, output_hidden_states):
        super(DIRT_T_model, self).__init__()
        self.bert = BertModel.from_pretrained("deepset/gbert-base")
        self.output_hidden_states = output_hidden_states
        self.feature_extractor = feature_extractor_module
        self.task_classifier = task_classifier_module
        self.domain_classifier = domain_classifier_module

    def forward(self, input_features):
        #bert_output = self.bert(input_ids=input_data[:,0], attention_mask=input_data[:,1], return_dict = False, output_hidden_states=self.output_hidden_states)
        feature_extractor_output = self.feature_extractor(input_features, input_is_bert = False)#(bert_output)

        class_output = self.task_classifier(feature_extractor_output)
        
        domain_output = self.domain_classifier(feature_extractor_output)

        return class_output, domain_output

    # inference for testing
    def inference(self, input_data):
        bert_output = self.bert(input_ids=input_data[:,0], attention_mask=input_data[:,1], return_dict = False, output_hidden_states=self.output_hidden_states)

        feature_extractor_output = self.feature_extractor(bert_output)

        class_output = self.task_classifier(feature_extractor_output)

        return class_output