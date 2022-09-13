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

        self.envpredictor = nn.Sequential()
        self.envpredictor.add_module('e_fc1', nn.Linear(in_features+1, 100))
        self.envpredictor.add_module('e_relu1', nn.ReLU())
        self.envpredictor.add_module('e_drop1', nn.Dropout())
        self.envpredictor.add_module('e_fc2', nn.Linear(100, 100))
        self.envpredictor.add_module('e_relu2', nn.ReLU())
        self.envpredictor.add_module('e_fc3', nn.Linear(100, 1))
        self.envpredictor.add_module('e_flatten', nn.Flatten(start_dim = 0))

    def forward(self, x, env):
        env_embedding = {
            'src':torch.zeros(x.size(0), 1).cuda(), 
            'tgt':torch.ones(x.size(0), 1).cuda()
        }

        x1 = torch.cat([x, env_embedding[env]], 1)

        x = self.envpredictor(x1)

        return x

class LIRR_model(nn.Module):
    def __init__(self, feature_extractor_module, domain_classifier, domain_invariant_predictor, bottleneck_dim):
        super(LIRR_model, self).__init__()
        self.bert = BertModel.from_pretrained("deepset/gbert-base")
        self.output_hidden_states = True
        self.feature_extractor = feature_extractor_module
        # TODO
        # specify in_features depending on last layer of feature_extractor

        self.domain_dependant_predictor = EnvPredictor(in_features = bottleneck_dim)
        self.domain_classifier = domain_classifier
        self.domain_invariant_predictor = domain_invariant_predictor

    def forward(self, input_data, env = 'src', reverse = False, eta = 1.0):
        bert_output = self.bert(input_ids=input_data[:,0], attention_mask=input_data[:,1], return_dict = False, output_hidden_states=self.output_hidden_states)
        
        feature_extractor_output = self.feature_extractor(bert_output)

        domain_dependant_output = self.domain_dependant_predictor(feature_extractor_output, env)

        if reverse == True:
            feature_extractor_output = ReverseLayerF.apply(feature_extractor_output, eta)

        domain_classifier_output = self.domain_classifier(feature_extractor_output)
        
        class_output = self.domain_invariant_predictor(feature_extractor_output)

        return domain_dependant_output, domain_classifier_output, class_output

    # inference for testing
    def inference(self, input_data):
        bert_output = self.bert(input_ids=input_data[:,0], attention_mask=input_data[:,1], return_dict = False, output_hidden_states=self.output_hidden_states)
        
        feature_extractor_output = self.feature_extractor(bert_output)

        class_output = self.domain_invariant_predictor(feature_extractor_output)

        return class_output