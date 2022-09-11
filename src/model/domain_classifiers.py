import torch
from torch import nn
from transformers import BertModel
from torch.nn import CrossEntropyLoss
import gc

class DANN_domain_classifier(nn.Module):
    def __init__(self, bottleneck_dim = 768, layer_size = 100):
        super().__init__()
        
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(bottleneck_dim, layer_size))
        self.domain_classifier.add_module('d_relu1', nn.ReLU())
        self.domain_classifier.add_module('d_drop1', nn.Dropout())
        self.domain_classifier.add_module('d_fc2', nn.Linear(layer_size, layer_size))
        self.domain_classifier.add_module('d_relu2', nn.ReLU())
        self.domain_classifier.add_module('d_fc2', nn.Linear(layer_size, 1))
        self.domain_classifier.add_module('d_flatten', nn.Flatten(start_dim = 0))

    def forward(
        self,
        feature_extractor_output
    ):
        x = self.domain_classifier(feature_extractor_output)
        
        return x

