import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from torch.multiprocessing import Pool, Process, set_start_method
import torch
#from model.get_model_alt import get_model
from torch.utils.tensorboard import SummaryWriter
import gc

import sys
def run_experiments(config):
    #log_file = Path(log_dir / "results.csv")

    for experiment in config["experiments"]:
        writer = SummaryWriter()

        basic_settings = experiment["basic_settings"]
        
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

            # exp_name = exp_setting.get("exp_name", "standard_name")

            # print(f"Performing training for: {exp_name}")

            exp_type = exp_setting["exp_type"]

            if exp_type == "DANN":
                from experiment_DANN import experiment_DANN
                current_exp = experiment_DANN(
                    basic_settings, exp_setting, writer
                )

            elif exp_type == "DIRT_T":
                from experiment_DIRT_T import experiment_DIRT_T
                current_exp = experiment_DIRT_T(
                    basic_settings, exp_setting, writer#, log_path, writer
                )
            # elif exp_type == "MME":
            #     current_exp = experiment_MME(
            #         basic_settings, exp_setting#, log_path, writer
            #     )
            # elif exp_type == "LIRR":
            #     current_exp = experiment_LIRR(
            #         basic_settings, exp_setting#, log_path, writer
            #     )
            # elif exp_type == "MDAN":
            #     current_exp = experiment_MDAN(
            #         basic_settings, exp_setting#, log_path, writer
            #     )
            # elif exp_type == "M3SDA":
            #     current_exp = experiment_M3SDA(
            #         basic_settings, exp_setting#, log_path, writer
            #     )
            current_exp.perform_experiment()
            # try:
            #     current_exp.perform_experiment()
            #     del current_exp
            #     gc.collect()

            # except Exception as e:
            #     name = exp_setting["exp_name"]
            #     print("\n\n")
            #     print("**********" * 12)
            #     print(f"Experiment {name} failed with Exception {e}")
            #     print("**********" * 12)
            #     print("\n\n")
            sys.exit(0)
            #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
            # model = get_model(exp_setting)

            # trainer = get_trainer(model, exp_setting)

            # trainer.train()

            # model.to(device)
            # model.train()
            ###### create dataloader

            # if exp_type == "unsupervised "
                # get unsupervised train dataloader(target_unlabeled, unlabelled_size, sources, validation_split, test_split)
                # get unsupervised test dataloader(target_labelled, validation_split, test_split)
                # do train unsupervised
                # train_loader = get_unsupervised_dataloader()
            # elif exp_type == "semi-supervised":
                # get semi-supervised dataloader
                # do semi-supervise
                # train_loader = get_semi_supervised_dataloader()
            # elif exp_type == "multi-source":
                # get multi-source dataloader
                # do multi-source
                # train_loader = get_multi_source_dataloader()
            # else:
                # error: please specify "exp_type", either unsupervised, semi-supervised or multisource in experiment settings



            ####### create optimizer
            # self.optimizer = get_optimizer(...)
            # self.optimizer = optim.SGD(
        #     [
        #         {"params": base_params},
        #         {"params": gen_odin_params, "weight_decay": 0.0},
        #     ],
        #     weight_decay=self.weight_decay,
        #     lr=self.lr,
        #     momentum=self.momentum,
        #     nesterov=self.nesterov,
        # )

            ###### create criterion
            # criterion = get_criterion(...)


            ######### train
            # for epoch in range(1, epochs + 1):
            #
            #       for batch_idx, (data, target) in enumerate(train_loader):
                        # 
                        # optimizer.zero_grad(set_to_none=True)
                        # task_output, domain_output = model(data).to(device)
                        # model_output = model(data).to(device)
                        # loss = criterion(model_output)

                        # loss.backward()
                        # optimizer.step()


            # if config["model"] == "DIRT_T"
                # if "VADA" == True:
                    # do VADA

            # ...

            


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

    # parser.add_argument(
    #     "-l",
    #     "--log",
    #     help="Log-instructions in json",
    #     type=str,
    #     default=Path("log_dir"),
    # )

    args = parser.parse_args()

    config = args.config
    #log_dir_parent = args.log
    #log_dir = args.log

    with open(config, mode="r", encoding="utf-8") as config_f:
        config = json.load(config_f)

    now = datetime.now()

    #log_dir = Path(log_dir_parent / f"experiments_{now.strftime('%H%M%S-%Y%m%d')}")
    #log_dir.mkdir(parents=True, exist_ok=True)

    #run_experiments(log_dir, config)
    run_experiments(config)

if __name__ == "__main__":
    # try:
    #     set_start_method('spawn')
    # except RuntimeError:
    #     pass
    
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    main()