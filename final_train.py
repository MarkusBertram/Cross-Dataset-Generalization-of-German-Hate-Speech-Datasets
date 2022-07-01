import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import torch
from model.get_model import get_model

def run_experiments(log_dir, config):
    log_file = Path(log_dir / "results.csv")

    for experiment in config["experiments"]:
        basic_settings = experiment["basic_settings"]
        
        # data settings
        target_labelled = basic_settings.get("target_labelled", "telegram_gold")
        target_unlabelled = basic_settings.get("target_unlabelled", "telegram_unlabeled")
        unlabelled_size = basic_settings.get("unlabelled_size", 200000)
        validation_split = basic_settings.get("validation_split", 0)
        test_split = basic_settings.get("test_split", 0.2)
        sources = basic_settings.get("sources", [
            "germeval2018", 
            "germeval2019",
            "hasoc2019",
            "hasoc2020",
            "ihs_labelled",
            "covid_2021"
        ])
        
        # training settings
        epochs = basic_settings.get("epochs", 10)
        batch_size = basic_settings.get("batch_size", 64)
        weight_decay = basic_settings.get("weight_decay", 0)
        lr = basic_settings.get("lr", 2e-5)
        momentum = basic_settings.get("momentum", 0)
        freeze_BERT_weights = basic_settings.get("freeze_BERT_weights", True)

        with open(
            log_file, "w", encoding="utf-8"
        ) as result_file:
            result_file.write(
                "exp_name,avg_train_acc,avg_train_loss,avg_test_acc,avg_test_loss\n"
            )

        for exp_setting in experiment["exp_settings"]:
            exp_name = exp_setting.get("exp_name", "standard_name")

            print(f"Performing training for: {exp_name}")

            exp_type = exp_setting.get("exp_type", "baseline")

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # feature_extractor = exp_settings.get("feature_extractor", "f1")
            # task_classifier = exp_settings.get("task_classifer", "c1")



            # alignment_component = exp_settings.get("alignment_component", "DANN")

            #model = get_model(feature_extractor, task_classifer, alignment_component)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare run of Domain Adaptation",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--config",
        help="Path to the config file for the experiment",
        type=str,
        default=Path("settings/experiment_settings.json"),
    )

    parser.add_argument(
        "-l",
        "--log",
        help="Log-instructions in json",
        type=str,
        default=Path("log_dir"),
    )

    args = parser.parse_args()

    config = args.config
    log_dir_parent = args.log

    with open(config, mode="r", encoding="utf-8") as config_f:
        config = json.load(config_f)

    now = datetime.now()

    log_dir = Path(log_dir_parent / f"experiments_{now.strftime('%H%M%S-%Y%m%d')}")
    log_dir.mkdir(parents=True, exist_ok=True)

    run_experiments(log_dir, config)

if __name__ == "__main__":
    main()