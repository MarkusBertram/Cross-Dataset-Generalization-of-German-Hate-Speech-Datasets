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
import torch.nn.functional as F
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

class MDAN_model(nn.Module):
    def __init__(self, feature_extractor_module, task_classifier_module, domain_classifier_module, output_hidden_states, num_src_domains):
        super(MDAN_model, self).__init__()
        self.bert = BertModel.from_pretrained("deepset/gbert-base")
        self.output_hidden_states = output_hidden_states
        self.feature_extractor = feature_extractor_module
        self.task_classifier = task_classifier_module
        self.num_src_domains = num_src_domains
        # Domain Classifiers
        self.domain_classifiers = nn.ModuleList([domain_classifier_module for _ in range(self.num_src_domains)])

        # Gradient reversal layer.
        #self.grls = [ReverseLayerF() for _ in range(self.num_src_domains)]

    def forward(self, src_input_data, tgt_input_data, alpha=1):
        """
        :param sinputs:     A list of k inputs from k source domains.
        :param tinputs:     Input from the target domain.
        :return:
        """
        sh_relu, th_relu = src_input_data, tgt_input_data

        # Source Feature Extractor Outputs:
        for i in range(self.num_src_domains):
            sh_relu[i] = self.bert(input_ids=sh_relu[i][:,0], attention_mask=sh_relu[i][:,1], return_dict = False, output_hidden_states=self.output_hidden_states)
            sh_relu[i] = self.feature_extractor(sh_relu[i])
        # Target Feature Extractor Outputs:
        th_relu = self.bert(input_ids=th_relu[:,0], attention_mask=th_relu[:,1], return_dict = False, output_hidden_states=self.output_hidden_states)
        th_relu = self.feature_extractor(th_relu)

        # Task Classification probabilities on k source domains.
        logprobs = []
        for i in range(self.num_src_domains):
            logprobs.append(self.task_classifier(sh_relu[i]))

        # Domain classification accuracies.
        sdomains, tdomains = [], []

        for i in range(self.num_src_domains):
            #sh_relu[i] = self.grls[i].apply(sh_relu[i], alpha)
            #th_relu_i = self.grls[i].apply(th_relu, alpha)

            sh_relu[i] = ReverseLayerF.apply(sh_relu[i], alpha)
            th_relu_i = ReverseLayerF.apply(th_relu, alpha)

            sdomains.append(self.domain_classifiers[i](sh_relu[i]))
            tdomains.append(self.domain_classifiers[i](th_relu_i))

        return logprobs, sdomains, tdomains

    def inference(self, input):

        bert_output = self.bert(input_ids=input[:,0], attention_mask=input[:,1], return_dict = False, output_hidden_states=self.output_hidden_states)
        feature_extractor_output = self.feature_extractor(bert_output)

        # Classification probability.
        logprobs = self.task_classifier(feature_extractor_output)
        return logprobs