from feature_extractors import bert_cls



def get_model(
    feature_extractor, 
    task_classifer, 
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
        return resnet20(
            num_classes=kwargs.get("num_classes", 10),
            similarity=kwargs.get("similarity", None),
        )
    
    elif model_name == "LOOC":
        return resnet20(**kwargs)
    
    elif model_name == "gram_resnet":
        return get_gram_resnet(num_classes=kwargs.get("num_classes", 10))
    
    else:
        raise ValueError(f"Model {model_name} not found")