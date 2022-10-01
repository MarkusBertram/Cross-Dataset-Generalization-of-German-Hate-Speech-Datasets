import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from torch.multiprocessing import Pool, Process, set_start_method
import torch
#from src.model.get_model_alt import get_model
from torch.utils.tensorboard import SummaryWriter
import gc
import random
import numpy as np
from pathlib import Path
import sys
def run_experiments(config):
    #log_file = Path(log_dir / "results.csv")

    for experiment in config["experiments"]:
        writer = SummaryWriter()

        basic_settings = experiment["basic_settings"]
        if "seed" in basic_settings:
            torch.manual_seed(basic_settings["seed"])
            random.seed(basic_settings["seed"])
            np.random.seed(basic_settings["seed"])

        for exp_setting in experiment["exp_settings"]:

            if exp_setting.get("perform_experiment", True):
                print(
                    f'\n\nINFO ---- Experiment {exp_setting["exp_type"]} is being performed.\n\n'
                )
            else:
                print(
                    f'\n\nINFO ---- Experiment {exp_setting["exp_type"]} is not being performed.\n\n'
                )
                continue

            exp_type = exp_setting["exp_type"]
            if exp_type == "target_only":
                from src.experiments.experiment_target_only import experiment_target_only
                current_exp = experiment_target_only(
                    basic_settings, exp_setting, writer
                )
            elif exp_type == "source_combined":
                from src.experiments.experiment_source_combined import experiment_source_combined
                current_exp = experiment_source_combined(
                    basic_settings, exp_setting, writer
                )
            elif exp_type == "single_source":
                from src.experiments.experiment_single_source import experiment_single_source
                current_exp = experiment_single_source(
                    basic_settings, exp_setting, writer
                )
            elif exp_type == "DANN":
                from src.experiments.experiment_DANN import experiment_DANN
                current_exp = experiment_DANN(
                    basic_settings, exp_setting, writer
                )

            elif exp_type == "DIRT_T":
                from src.experiments.experiment_DIRT_T import experiment_DIRT_T
                current_exp = experiment_DIRT_T(
                    basic_settings, exp_setting, writer#, log_path, writer
                )
            elif exp_type == "MME":
                from src.experiments.experiment_MME import experiment_MME
                current_exp = experiment_MME(
                    basic_settings, exp_setting, writer#, log_path, writer
                )
            elif exp_type == "LIRR":
                from src.experiments.experiment_LIRR import experiment_LIRR
                current_exp = experiment_LIRR(
                    basic_settings, exp_setting, writer#, log_path, writer
                )
            elif exp_type == "MDAN":
                from src.experiments.experiment_MDAN import experiment_MDAN
                current_exp = experiment_MDAN(
                    basic_settings, exp_setting, writer#, log_path, writer
                )
            elif exp_type == "M3SDA":
                from src.experiments.experiment_M3SDA import experiment_M3SDA
                current_exp = experiment_M3SDA(
                    basic_settings, exp_setting, writer#, log_path, writer
                )
            current_exp.perform_experiment()     

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

    args = parser.parse_args()

    config = args.config

    with open(config, mode="r", encoding="utf-8") as config_f:
        config = json.load(config_f)

    run_experiments(config)

if __name__ == "__main__":
    
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    main()