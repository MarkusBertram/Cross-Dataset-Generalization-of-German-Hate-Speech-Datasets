import torch
from torch import nn
from transformers import BertModel
from torch.nn import CrossEntropyLoss
import gc

class domain_classifier1(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_labels = 2
        
        self.domain_classifier_linear1 = nn.Linear(768, 768)
        self.domain_classifier_linear2 = nn.Linear(768, 768)
        self.domain_classifier_linear3 = nn.Linear(768, 2)
        self.domain_classifier_relu = nn.LeakyReLU()
        self.domain_classifier_softmax = nn.Softmax(dim=1)
        self.domain_classifier_dropout = nn.Dropout(0.1)

    def domain_classifier_forward(
        self,
        feature_extractor_output
    ):
        x = self.domain_classifier_linear1(feature_extractor_output)

