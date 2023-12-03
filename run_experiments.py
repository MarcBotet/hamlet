import argparse
import json
import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
import shutil

import torch
from experiments import generate_experiment_cfgs

from experiments_custom import generate_experiment_cfgs as generate_experiment_cfgs_custom
CUSTOM = True

from mmcv import Config, get_git_hash

# from mmseg.apis import set_random_seed
from tools import train

DEBUG = False


def run_command(command):
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
    )
    for line in iter(p.stdout.readline, b""):
        print(line.decode("utf-8"), end="")


def rsync(src, dst):
    rsync_cmd = f"rsync -a {src} {dst}"
    print(rsync_cmd)
    run_command(rsync_cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--exp",
        type=int,
        nargs="*",
        default=None,
        help="Experiment id as defined in experiment.py",
    )
    group.add_argument(
        "--config",
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--wandb",
        type=int,
        default=0,
        help="1 to log results on wandb",
    )
    parser.add_argument(
        "--custom",
        type=int,
        default=1,
        help="activate CUSTOM mode"
    )

    parser.add_argument("--machine", type=str, choices=["local"], default="local")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    assert (args.config is None) != (
        args.exp is None
    ), "Either config or exp has to be defined."

    GEN_CONFIG_DIR = "configs/generated/"
    JOB_DIR = "jobs"
    cfgs, config_files = [], []

    # Training with Predefined Config
    if args.config is not None:
        cfg = Config.fromfile(args.config)
        # Specify Name and Work Directory
        exp_name = f'{args.machine}-{cfg["exp"]}'
        unique_name = (
            f'{datetime.now().strftime("%y%m%d_%H%M")}_'
            f'{cfg["name"]}_{str(uuid.uuid4())[:5]}'
        )
        child_cfg = {
            "_base_": args.config.replace("configs", "../.."),
            "name": unique_name,
            "work_dir": os.path.join("work_dirs", exp_name, unique_name),
            "git_rev": get_git_hash(),
        }
        cfg_out_file = f"{GEN_CONFIG_DIR}/{exp_name}/{child_cfg['name']}.json"
        os.makedirs(os.path.dirname(cfg_out_file), exist_ok=True)
        assert not os.path.isfile(cfg_out_file)
        with open(cfg_out_file, "w") as of:
            json.dump(child_cfg, of, indent=4)
        config_files.append(cfg_out_file)
        cfgs.append(cfg)

    # Training with Generated Configs from experiments.py
    if args.exp is not None:
        for exp in args.exp:
            exp_name = f"{args.machine}-exp{exp}"

            if CUSTOM:
                cfgs_aux = generate_experiment_cfgs_custom(exp)
            else:
                cfgs_aux = generate_experiment_cfgs(exp)

            # Generate Configs
            for i, cfg in enumerate(cfgs_aux):
                if args.debug:
                    cfg.setdefault("log_config", {})["interval"] = 10
                    cfg["evaluation"] = dict(interval=200, metric="mIoU")
                    if "dacs" in cfg["name"]:
                        cfg.setdefault("uda", {})["debug_img_interval"] = 10
                        # cfg.setdefault('uda', {})['print_grad_magnitude'] = True
                # Generate Config File
                cfg["name"] = (
                    f'{datetime.now().strftime("%y%m%d_%H%M")}_'
                    f'{cfg["name"]}_{str(uuid.uuid4())[:5]}'
                )
                if DEBUG:
                    cfg["name"] = f"debug_{cfg['name']}"
                cfg["work_dir"] = os.path.join("work_dirs", exp_name, cfg["name"])
                cfg["git_rev"] = get_git_hash()
                tags = cfg["tags"].copy()
                del cfg["tags"]
                epoch = "online" in cfg["mode"]
                if args.wandb:
                    notes = tags[0]
                    cfg["log_config"] = dict(
                        interval=1,
                        hooks=[
                            dict(type="TextLoggerHook", by_epoch=epoch),
                            dict(
                                type="WandbLoggerHook",
                                by_epoch=epoch,
                                init_kwargs=dict(
                                    project=cfg["wandb_project"],
                                    name=cfg["name_mine"],
                                    config=cfg.copy(),
                                    tags=tags,
                                    notes=notes,
                                ),
                            ),
                        ],
                    )
                else:
                    cfg["log_config"] = dict(
                        interval=1,
                        hooks=[
                            dict(type="TextLoggerHook", by_epoch=epoch),
                        ],
                    )

                cfg["_base_"] = ["../../" + e for e in cfg["_base_"]]
                cfg_out_file = f"{GEN_CONFIG_DIR}/{exp_name}/{cfg['name']}.json"
                os.makedirs(os.path.dirname(cfg_out_file), exist_ok=True)
                assert not os.path.isfile(cfg_out_file)
                with open(cfg_out_file, "w") as of:
                    json.dump(cfg, of, indent=4)
                config_files.append(cfg_out_file)
            cfgs.append(cfgs_aux)

    if args.machine == "local":
        i = 0
        for cfgs_aux in cfgs:
            for cfg in cfgs_aux:
                print("Run job {}".format(cfg["name"]))
                print(f"job number {i}")
                train.main([config_files[i]])
                torch.cuda.empty_cache()
                i += 1
    else:
        raise NotImplementedError(args.machine)
