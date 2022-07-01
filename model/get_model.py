def get_model(
    model_name,
    **kwargs,
):
    """get_model [[function which returns instance of the experiments model]]
    [extended_summary]
    Args:
        model_name ([string]): ["base":conv_net, "gen_odin_conv":conv_net with GenOdin,
            "gen_odin_res": large resnet with GenOdin Layer, "small_gen_odin_res": small resnet with GenOdin Layer,
            "small_resnet_with_spec": small resnet with spectral normalization, "base_small_resnet": abdurs resnet (working the best as per usual)  ]
        similarity ([type], optional): [For genOdinMode "E":Euclidean distance, "I": FC Layer, "C": Cosine Similarity, "IR": I-reversed , "ER": E-revesered, "CR": "C-reversed" ]. Defaults to None.
        num_classes (int, optional): [Number of classes]. Defaults to 10.
        include_bn (bool, optional): [Include batchnorm]. Defaults to False.
        channel_input (int, optional): [dataset channel]. Defaults to 3.
    Raises:
        ValueError: [When model name false]
    Returns:
        [nn.Module]: [parametrized Neural Network]
    """
    if model_name == "base":
        return resnet20(
            num_classes=kwargs.get("num_classes", 10),
            similarity=kwargs.get("similarity", None),
        )
    elif model_name == "GenOdin":
        return resnet20(
            num_classes=kwargs.get("num_classes", 10),
            similarity=kwargs.get("similarity", "CR"),
        )
    elif model_name == "LOOC":
        return resnet20(**kwargs)
    elif model_name == "DDU":
        return resnet_ddu(
            num_classes=kwargs.get("num_classes", 10),
            spectral_normalization=kwargs.get("spectral_normalization", True),
            temp=kwargs.get("temp", 1.0),
        )
    elif model_name == "gram_resnet":
        return get_gram_resnet(num_classes=kwargs.get("num_classes", 10))
    elif model_name == "maximum_discrepancy":
        return resnet20(
            num_classes=kwargs.get("num_classes", 10),
            similarity=None,
            maximum_discrepancy=kwargs.get("maximum_discrepancy", True),
        )
    else:
        raise ValueError(f"Model {model_name} not found")