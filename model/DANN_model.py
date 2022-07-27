#from feature_extractors import BERT_cnn, bert_cls
from pyrsistent import s
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, BatchEncoding, Trainer, TrainingArguments, AdamW
#from functions import ReverseLayerF
import torch
from torch import nn
from transformers import BertModel
from torch.nn import CrossEntropyLoss
import gc
import sys

from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class DANN_model(nn.Module):
    def __init__(self, feature_extractor_module, task_classifier_module, domain_classifier_module, output_hidden_states):
        super(DANN_model, self).__init__()
        self.bert = BertModel.from_pretrained("deepset/gbert-base")
        self.output_hidden_states = output_hidden_states
        self.feature_extractor = feature_extractor_module
        self.task_classifier = task_classifier_module
        self.domain_classifier = domain_classifier_module

    def forward(self, input_data, alpha):
        bert_output = self.bert(input_ids=input_data[:,0], attention_mask=input_data[:,1], return_dict = False, output_hidden_states=self.output_hidden_states)
        
        feature_extractor_output = self.feature_extractor(bert_output)

        class_output = self.task_classifier(feature_extractor_output)
        
        reverse_feature = ReverseLayerF.apply(feature_extractor_output, alpha)

        domain_output = self.domain_classifier(reverse_feature)
    
        return class_output, domain_output

    # inference for testing
    def inference(self, input_data):
        bert_output = self.bert(input_ids=input_data[:,0], attention_mask=input_data[:,1], return_dict = False, output_hidden_states=self.output_hidden_states)
        
        feature_extractor_output = self.feature_extractor(bert_output)

        class_output = self.task_classifier(feature_extractor_output)

        return class_output