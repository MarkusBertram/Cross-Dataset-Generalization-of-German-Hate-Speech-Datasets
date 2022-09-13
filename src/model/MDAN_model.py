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

class ReverseLayerF(Function):

    @staticmethod
    def forward(self, x):

        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = -grad_input

        return grad_input

class MDAN_model(nn.Module):
    def __init__(self, feature_extractor_module, task_classifier_module, domain_classifier_module, num_src_domains):
        super(MDAN_model, self).__init__()
        self.bert = BertModel.from_pretrained("deepset/gbert-base")
        self.output_hidden_states = True
        self.feature_extractor = feature_extractor_module
        self.task_classifier = task_classifier_module
        self.num_src_domains = num_src_domains
        # Domain Classifiers
        self.domain_classifiers = nn.ModuleList([domain_classifier_module for _ in range(self.num_src_domains)])

        # Gradient reversal layer.
        #self.grls = [ReverseLayerF() for _ in range(self.num_src_domains)]

    def forward(self, src_input_data, tgt_input_data):
        """
        :param sinputs:     A list of k inputs from k source domains. [k, batch_size, 2, 512]
        :param tinputs:     Input from the target domain.
        :return:
        """
        # Source Feature Extractor Outputs:
        source_features = []
        for src_dset_batch in src_input_data:
            source_bert_output = self.bert(input_ids=src_dset_batch[:,0], attention_mask=src_dset_batch[:,1], return_dict = False, output_hidden_states=self.output_hidden_states)
            source_feature_output = self.feature_extractor(source_bert_output)
            source_features.append(source_feature_output)

        source_feature_outputs = torch.stack(source_features)
        
        # Target Feature Extractor Outputs:
        target_features = self.bert(input_ids=tgt_input_data[:,0], attention_mask=tgt_input_data[:,1], return_dict = False, output_hidden_states=self.output_hidden_states)
        target_features = self.feature_extractor(target_features)

        # Task Classification probabilities on k source domains.
        class_probabilities = torch.stack([self.task_classifier(source_feature) for source_feature in source_feature_outputs])

        # # Domain classification accuracies.
        # sdomains, tdomains = [], []
        
        # for i in range(self.num_src_domains):
        #     #sh_relu[i] = self.grls[i].apply(sh_relu[i], alpha)
        #     #th_relu_i = self.grls[i].apply(th_relu, alpha)

        #     reverse_src_feature = ReverseLayerF.apply(source_feature_outputs[i], alpha)
        #     reverse_target_feature = ReverseLayerF.apply(target_features, alpha)

        #     sdomains.append(self.domain_classifiers[i](sh_relu[i]))
        #     tdomains.append(self.domain_classifiers[i](th_relu_i))
        
        sdomains = torch.stack([self.domain_classifiers[i](ReverseLayerF.apply(source_feature_outputs[i])) for i in range(self.num_src_domains)])
        tdomains = torch.stack([self.domain_classifiers[i](ReverseLayerF.apply(target_features)) for i in range(self.num_src_domains)])
        
        return class_probabilities, sdomains, tdomains

    # inference for testing
    def inference(self, input_data):
        bert_output = self.bert(input_ids=input_data[:,0], attention_mask=input_data[:,1], return_dict = False, output_hidden_states=self.output_hidden_states)
        
        feature_extractor_output = self.feature_extractor(bert_output)

        class_output = self.task_classifier(feature_extractor_output)

        return class_output