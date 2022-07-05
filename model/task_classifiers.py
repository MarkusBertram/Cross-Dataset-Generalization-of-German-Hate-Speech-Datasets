import torch
from torch import nn
from transformers import BertModel
from torch.nn import CrossEntropyLoss
import gc

class task_classifier1(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_labels = 2
        
        # self.task_classifier_linear1 = nn.Linear(768, 768)
        # self.task_classifier_linear2 = nn.Linear(768, 768)
        # self.task_classifier_linear3 = nn.Linear(768, 2)
        # self.task_classifier_relu = nn.LeakyReLU()
        # self.task_classifier_softmax = nn.Softmax(dim=1)
        # self.task_classifier_dropout = nn.Dropout(0.1)

        class_classifier = nn.Sequential()
        class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        class_classifier.add_module('c_relu1', nn.ReLU(True))
        class_classifier.add_module('c_drop1', nn.Dropout2d())
        class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        class_classifier.add_module('c_relu2', nn.ReLU(True))
        class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        class_classifier.add_module('c_softmax', nn.LogSoftmax())


    def task_classifier_forward(
        self,
        feature_extractor_output
    ):
        #x = self.task_classifier_linear1(feature_extractor_output)

        x = self.class_classifier(feature_extractor_output)

        return x