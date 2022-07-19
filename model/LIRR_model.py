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

class EnvPredictor(nn.Module):
    def __init__(self, in_features, bottleneck_dim=1024):
        super(EnvPredictor, self).__init__()
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(in_features=in_features+1, out_features=bottleneck_dim, bias=True)
        self.fc = nn.Linear(in_features=bottleneck_dim, out_features=2, bias=True)
        
        #self.apply(init_weights)
        self.num_class = 2
        # self.env_embedding = {
        #     'src':torch.zeros(batch_size, 1).cuda(), 
        #     'tgt':torch.ones(batch_size, 1).cuda()
        # }

    def forward(self, x, env, temp=1, cosine=False):
        env_embedding = {
            'src':torch.zeros(x.size(0), 1).cuda(), 
            'tgt':torch.ones(x.size(0), 1).cuda()
        }

        x1 = torch.cat([x, env_embedding[env]], 1)
        drop_x = self.dropout2(x1)
        encodes = torch.nn.functional.relu(self.bottleneck(drop_x), inplace=False)
        drop_x = self.dropout2(encodes)
        if cosine:
    #       cosine classifer
            normed_x = nn.functional.normalize(drop_x, p=2, dim=1)
            logits = self.fc(normed_x) / temp
        else:
            logits = self.fc(drop_x) / temp
        return logits

class LIRR_model(nn.Module):
    def __init__(self, feature_extractor_module, domain_classifier, domain_invariant_predictor, output_hidden_states):
        super(LIRR_model, self).__init__()
        self.bert = BertModel.from_pretrained("deepset/gbert-base")
        self.output_hidden_states = output_hidden_states
        self.feature_extractor = feature_extractor_module
        # TODO
        # specify in_features depending on last layer of feature_extractor

        self.domain_dependant_predictor = EnvPredictor(in_features = 768)
        self.domain_classifier = domain_classifier
        self.domain_invariant_predictor = domain_invariant_predictor

        #self.task_classifier = task_classifier_module

    def forward(self, input_data, env = 'src', reverse = False, eta = 0.1):#, #eta = 0.1):
        bert_output = self.bert(input_ids=input_data[:,0], attention_mask=input_data[:,1], return_dict = False, output_hidden_states=self.output_hidden_states)
        
        feature_extractor_output = self.feature_extractor(bert_output)

        domain_dependant_output = self.domain_dependant_predictor(feature_extractor_output, env)

        if reverse == True:
            feature_extractor_output = ReverseLayerF.apply(feature_extractor_output, eta)
            # feature_extractor_output = ## Relu?
            ### Relu?
            ## bottleneck?

        domain_classifier_output = self.domain_classifier(feature_extractor_output)
        class_output = self.domain_invariant_predictor(feature_extractor_output)
        
        # class_output = self.task_classifier(feature_extractor_output)

        return domain_dependant_output, domain_classifier_output, class_output