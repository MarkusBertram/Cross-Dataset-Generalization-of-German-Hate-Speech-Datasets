import torch
from torch import nn
from transformers import BertModel
from torch.nn import CrossEntropyLoss
import gc

class DANN_task_classifier(nn.Module):
    def __init__(self, bottleneck_dim = 768, layer_size = 100):
        super().__init__()

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(bottleneck_dim, layer_size))
        self.class_classifier.add_module('c_relu1', nn.ReLU())
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(layer_size, layer_size))
        self.class_classifier.add_module('c_relu2', nn.ReLU())
        self.class_classifier.add_module('c_fc3', nn.Linear(layer_size, 1))
        self.class_classifier.add_module('c_flatten', nn.Flatten(start_dim = 0))
        

    def forward(
        self,
        feature_extractor_output
    ):
    
        x = self.class_classifier(feature_extractor_output)

        return x