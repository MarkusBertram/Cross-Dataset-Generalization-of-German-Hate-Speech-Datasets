import torch
from torch import nn
import gc

class BERT_cnn(nn.Module):
    def __init__(self, bottleneck_dim = 768):
        super(BERT_cnn, self).__init__()
        # truncation length = 512
        self.feature_extractor_conv = nn.Conv2d(in_channels=13, out_channels=13, kernel_size=(3, 768), padding = (1,0))
        self.feature_extractor_relu = nn.ReLU()
        self.feature_extractor_pool = nn.MaxPool1d(kernel_size=3)
        self.feature_extractor_fc = nn.Linear(2210, bottleneck_dim)
        self.feature_extractor_dropout = nn.Dropout(0.1)
        self.feature_extractor_flat = nn.Flatten()
       
    def forward(self, bert_output, input_is_bert = True):
        if input_is_bert:
            x = torch.swapaxes(torch.stack(bert_output[2]), 0, 1)
        else:
            x = bert_output
        
        x = self.feature_extractor_dropout(x)

        x = self.feature_extractor_conv(x)
        x = x.squeeze()
        x = self.feature_extractor_relu(x)
        x = self.feature_extractor_dropout(x)
        x = self.feature_extractor_pool(x)
        x = self.feature_extractor_dropout(x)
        if x.size(dim=0) == 13 and x.size(dim=1) == 170:
            x = x.unsqueeze(0)
        x = self.feature_extractor_flat(x)
        x = self.feature_extractor_dropout(x)
        x = self.feature_extractor_fc(x)
        return x

