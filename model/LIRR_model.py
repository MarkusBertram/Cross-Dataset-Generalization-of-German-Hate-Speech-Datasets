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

# Class EnvPredictor(...)

class LIRR_model(nn.Module):
    def __init__(self, feature_extractor_module, domain_dependant_predictor, domain_classifier, domain_invariant_predictor, output_hidden_states):
        super(LIRR_model, self).__init__()
        self.bert = BertModel.from_pretrained("deepset/gbert-base")
        self.output_hidden_states = output_hidden_states
        self.feature_extractor = feature_extractor_module

        self.domain_dependant_predictor = domain_dependant_predictor
        self.domain_classifier = domain_classifier
        self.domain_invariant_predictor = domain_invariant_predictor

        #self.task_classifier = task_classifier_module

    def forward(self, input_data, reverse = False, eta = 0.1):
        bert_output = self.bert(input_ids=input_data[:,0], attention_mask=input_data[:,1], return_dict = False, output_hidden_states=self.output_hidden_states)
        
        feature_extractor_output = self.feature_extractor(bert_output)

        domain_dependant_output = self.domain_dependant_predictor(feature_extractor_output)

        if reverse == True:
            feature_extractor_output = ReverseLayerF.apply(feature_extractor_output, eta)
            # feature_extractor_output = ## Relu?
            ### Relu?
            ## bottleneck?

        domain_classifier_output = self.domain_classifier(feature_extractor_output)
        class_output = self.domain_invariant_predictor

        # class_output = self.task_classifier(feature_extractor_output)

        return domain_dependant_output, domain_classifier_output, class_output