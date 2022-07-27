import torch
from torch import nn
from transformers import BertModel
from torch.nn import CrossEntropyLoss
import gc

class DANN_domain_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_labels = 2
        
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(768, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 1))
        self.domain_classifier.add_module('d_flatten', nn.Flatten(start_dim = 0))
        #self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))

    def forward(
        self,
        feature_extractor_output
    ):
        x = self.domain_classifier(feature_extractor_output)
        
        return x

