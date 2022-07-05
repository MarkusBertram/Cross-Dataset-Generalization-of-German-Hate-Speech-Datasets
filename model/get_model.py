#from feature_extractors import BERT_cnn, bert_cls
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, BatchEncoding, Trainer, TrainingArguments, AdamW

import torch
from torch import nn
from transformers import BertModel
from torch.nn import CrossEntropyLoss
import gc
import sys

def get_model(
    feature_extractor, 
    task_classifier,
    domain_classifier, 
    alignment_component,
    **kwargs,
):
    """get_model [[function which returns instance of the experiments model]]
    [extended_summary]
    Args:
        feature_extractor ([string]): ["bert_cls":bert with cls token, "bert_cnn": bert with cnn].
        task_classifier ([string]): ["fd1": X hidden layers]

        similarity ([type], optional): [For genOdinMode "E":Euclidean distance, "I": FC Layer, "C": Cosine Similarity, "IR": I-reversed , "ER": E-revesered, "CR": "C-reversed" ]. Defaults to None.
        num_classes (int, optional): [Number of classes]. Defaults to 10.
        include_bn (bool, optional): [Include batchnorm]. Defaults to False.
        channel_input (int, optional): [dataset channel]. Defaults to 3.
    Raises:
        ValueError: [When model name false]
    Returns:
        [nn.Module]: [parametrized Neural Network]
    """
    # feature_extractor = get_feature_extractor(feature_extractor_name)

    # task_classifier = get_task_classifer(task_classifier_name)

    if feature_extractor == "bert_cls":
        from feature_extractors import BERT_cls
        feature_extractor_module = BERT_cls()
        output_hidden_states = False
    elif feature_extractor == "bert_cnn":
        from feature_extractors import BERT_cnn
        feature_extractor_module = BERT_cnn()
        output_hidden_states = True

    if task_classifier == "tc1":
        # from task_classifiers import task_classifier1
        # task_classifier_module = task_classifier1()
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
    
    if domain_classifier == "dc1":
        from domain_classifiers import domain_classifier1
        domain_classifier_module = domain_classifier1()
    
    print(feature_extractor_module)
    print("aaa")
    #     return resnet20(
    #         num_classes=kwargs.get("num_classes", 10),
    #         similarity=kwargs.get("similarity", None),
    #     )
    
    # elif model_name == "LOOC":
    #     return resnet20(**kwargs)
    
    # elif model_name == "gram_resnet":
    #     return get_gram_resnet(num_classes=kwargs.get("num_classes", 10))
    
    # else:
    #     raise ValueError(f"Model {model_name} not found")

    class MultiHeadNetwork(nn.Module):
        def __init__(self):
            super(MultiHeadNetwork, self).__init__()
            self.bert = BertModel.from_pretrained("deepset/gbert-base")
            self.output_hidden_states = output_hidden_states
            self.feature_extractor = feature_extractor_module
            #self.task_classifier = task_classifier_module
            self.task_classifier = class_classifier
            self.domain_classifier = domain_classifier_module

        def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):
            bert_output = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=self.output_hidden_states)
            feature_extractor_output = self.feature_extractor(bert_output)

            class_output = self.task_classifier(feature_extractor_output)
            domain_output = self.domain_classifier(feature_extractor_output)


            # if reverse_gradient == True:
            # do reverse 

            # if ... == True:
            # do ...
            return class_output, domain_output



    return MultiHeadNetwork

model = get_model("bert_cls", "tc1", "dc1", None)
model = model()
print(model())