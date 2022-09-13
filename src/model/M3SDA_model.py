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
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class M3SDA_model(nn.Module):
    def __init__(self, feature_extractor_module, task_classifier_module, num_src_domains):
        super(M3SDA_model, self).__init__()
        self.bert = BertModel.from_pretrained("deepset/gbert-base")
        self.output_hidden_states = True
        self.feature_extractor = feature_extractor_module
        #self.task_classifier = task_classifier_module
        self.num_src_domains = num_src_domains
        # Domain Classifiers
        self.task_classifiers = nn.ModuleList([nn.ModuleList([task_classifier_module, task_classifier_module]) for _ in range(self.num_src_domains)])

        self.weights = np.empty(self.num_src_domains)

    def forward(self, input_data, index = None, feature_extractor_input = False, output_only_features = False, reverse = False, alpha=1):
        """
        :src_input_data:     A list of inputs from k source domains.
        :tgt_input_data:     Input from the target domain.
        :return:
        """
        if feature_extractor_input == False:
            bert_output = self.bert(input_ids=input_data[:,0], attention_mask=input_data[:,1], return_dict = False, output_hidden_states=self.output_hidden_states)
            feature_extractor_output = self.feature_extractor(bert_output)
        else:
            feature_extractor_output = input_data
        
        if output_only_features == True:
            return feature_extractor_output

        if reverse:
            feature_extractor_output = ReverseLayerF.apply(feature_extractor_output, alpha)

        
        pred1 = self.task_classifiers[index][0](feature_extractor_output)
        pred2 = self.task_classifiers[index][1](feature_extractor_output)

        return pred1, pred2

    def inference(self, input):

        bert_output = self.bert(input_ids=input[:,0], attention_mask=input[:,1], return_dict = False, output_hidden_states=self.output_hidden_states)
        feature_extractor_output = self.feature_extractor(bert_output)

        prediction_list = []
        
        # Classification probability.
        for i in range(self.num_src_domains):
            pred1 = self.task_classifiers[i][0](feature_extractor_output)
            pred2 = self.task_classifiers[i][1](feature_extractor_output)
            avg_pred = (pred1 + pred2) / 2.0
            prediction_list.append(avg_pred)

        final_pred = torch.zeros_like(prediction_list[0])
        for i in range(self.num_src_domains):
            final_pred += self.weights[i] * prediction_list[i]
        return final_pred