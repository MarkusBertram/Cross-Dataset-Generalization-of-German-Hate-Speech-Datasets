import torch
from torch import nn
from transformers import BertModel
from torch.nn import CrossEntropyLoss
import gc

class task_classifier1(nn.Module):
    def __init__(self):
        self.num_labels = 2
        
        self.task_classifier_linear1 = nn.Linear(768, 768)
        self.task_classifier_linear2 = nn.Linear(768, 768)
        self.task_classifier_linear3 = nn.Linear(768, 2)
        self.task_classifier_relu = nn.LeakyReLU()
        self.task_classifier_softmax = nn.Softmax(dim=1)
        self.task_classifier_dropout = nn.Dropout(0.1)


    def task_classifier_forward(
        self,
        feature_extractor_output
    ):
        x = self.task_classifier_linear1(feature_extractor_output)