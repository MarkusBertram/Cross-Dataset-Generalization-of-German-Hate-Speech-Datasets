from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, BatchEncoding, Trainer, TrainingArguments, AdamW

import torch
from torch import nn
from transformers import BertModel
from torch.nn import CrossEntropyLoss
import gc

class BERT_cls(nn.Module):
    def __init__(self):
        super(BERT_cls, self).__init__()
        self.num_labels = 2
        # self.bert = BertModel.from_pretrained("deepset/gbert-base")
        ### New layers:
        self.feature_extractor_linear1 = nn.Linear(768, 768)
        self.feature_extractor_linear2 = nn.Linear(768, 768)
        self.feature_extractor_linear3 = nn.Linear(768, 768)
        self.feature_extractor_relu = nn.LeakyReLU()
        # self.feature_extractor_softmax = nn.Softmax(dim=1)
        self.feature_extractor_dropout = nn.Dropout(0.1)
       
    def forward(self, bert_output, input_is_bert = True):
        if input_is_bert == True:
            cls_token = bert_output[0][:,0,:]
        else:
            cls_token = bert_output
        #x = self.feature_extractor_dropout(cls_token)
        x = self.feature_extractor_linear1(cls_token) 
        x = self.feature_extractor_relu(x)
        x = self.feature_extractor_dropout(x)
        x = self.feature_extractor_linear2(x)
        x = self.feature_extractor_relu(x)
        x = self.feature_extractor_dropout(x)
        x = self.feature_extractor_linear3(x)
        x = self.feature_extractor_relu(x)
        x = self.feature_extractor_dropout(x)
        return x


class BERT_cnn(nn.Module):
    def __init__(self):
        super(BERT_cnn, self).__init__()
        self.num_labels = 2
        # self.feature_extractor_bert = BertModel.from_pretrained("deepset/gbert-base")
        ### New layers:
        self.feature_extractor_conv = nn.Conv2d(in_channels=13, out_channels=13, kernel_size=(3, 768), padding=True)
        self.feature_extractor_relu = nn.ReLU()
        self.feature_extractor_pool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.feature_extractor_dropout = nn.Dropout(0.1)
        self.feature_extractor_fc = nn.Linear(442, 3)
        self.feature_extractor_flat = nn.Flatten()
        self.feature_extractor_softmax = nn.LogSoftmax(dim=1)
       
    def forward(
        self,
        cls_token
        ):
        #_, _, all_layers = self.feature_extractor_bert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # all_layers  = [13, 32, 64, 768]
        #x = torch.transpose(torch.cat(tuple([t.unsqueeze(0) for t in all_layers]), 0), 0, 1)
        del all_layers
        gc.collect()
        torch.cuda.empty_cache()
        x = self.feature_extractor_dropout(x)
        x = self.feature_extractor_conv(x)
        x = self.feature_extractor_relu(x)
        x = self.feature_extractor_dropout(x)
        x = self.feature_extractor_pool(x)
        
        x = self.feature_extractor_dropout(x)
        x = self.feature_extractor_flat(x)
        x = self.feature_extractor_dropout(x)
        x = self.feature_extractor_fc(x)
        return self.feature_extractor_softmax(x)


class BERT_cnn_token_input(nn.Module):
    def __init__(self):
        super(BERT_cnn, self).__init__()
        self.num_labels = 2
        # self.feature_extractor_bert = BertModel.from_pretrained("deepset/gbert-base")
        ### New layers:
        self.feature_extractor_conv = nn.Conv2d(in_channels=13, out_channels=13, kernel_size=(3, 768), padding=True)
        self.feature_extractor_relu = nn.ReLU()
        self.feature_extractor_pool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.feature_extractor_dropout = nn.Dropout(0.1)
        self.feature_extractor_fc = nn.Linear(442, 3)
        self.feature_extractor_flat = nn.Flatten()
        self.feature_extractor_softmax = nn.LogSoftmax(dim=1)
       
    def forward(
        self,
        bert_output
        ):
        #_, _, all_layers = self.feature_extractor_bert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # all_layers  = [13, 32, 64, 768]
        x = torch.transpose(torch.cat(tuple([t.unsqueeze(0) for t in all_layers]), 0), 0, 1)
        del all_layers
        gc.collect()
        torch.cuda.empty_cache()
        x = self.feature_extractor_dropout(x)
        x = self.feature_extractor_conv(x)
        x = self.feature_extractor_relu(x)
        x = self.feature_extractor_dropout(x)
        x = self.feature_extractor_pool(x)
        
        x = self.feature_extractor_dropout(x)
        x = self.feature_extractor_flat(x)
        x = self.feature_extractor_dropout(x)
        x = self.feature_extractor_fc(x)
        return self.feature_extractor_softmax(x)

