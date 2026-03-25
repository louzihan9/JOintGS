#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import glob
import json
import os
import subprocess
import sys
import time
import argparse
from loguru import logger
from omegaconf import OmegaConf

sys.path.append('.')
import torch
from jointgs.trainer import GaussianTrainer, GaussianOptimTrainer
from jointgs.utils.config import get_cfg_items
from jointgs.cfg.config import cfg as default_cfg
from jointgs.utils.general import safe_state, find_cfg_diff

os.environ['TORCH_HOME'] = './data'


def get_logger(cfg):
    output_path = cfg.output_path
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    mode = 'eval' if cfg.eval else 'train'

    logdir = os.path.join(
        output_path, 'human_scene', cfg.dataset.name,
        cfg.dataset.seq, cfg.exp_name,
        time_str,
    )

    cfg.logdir = logdir
    cfg.logdir_ckpt = os.path.join(logdir, 'ckpt')

    os.makedirs(logdir, exist_ok=True)
    os.makedirs(cfg.logdir_ckpt, exist_ok=True)
    os.makedirs(os.path.join(logdir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(logdir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(logdir, 'anim'), exist_ok=True)
    os.makedirs(os.path.join(logdir, 'optim'), exist_ok=True)
    os.makedirs(os.path.join(logdir, 'meshes'), exist_ok=True)

    logger.add(os.path.join(logdir, f'{mode}.log'), level='INFO')

    with open(os.path.join(logdir, f'config_{mode}.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))


def param_optim(cfg):
    safe_state(seed=cfg.seed)
    cfg.if_refine = False
    trainer = GaussianOptimTrainer(cfg)
    trainer.train()


def only_train(cfg, use_refine_data=True):
    safe_state(seed=cfg.seed)
    cfg.if_refine = use_refine_data
    trainer = GaussianTrainer(cfg)
    trainer.train()



if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg_file", default = 'cfg/neuman/human_scene.yaml', help="path to the yaml config file")
    parser.add_argument("--cfg_id", type=int, default=-1, help="id of the config to run")
    args, extras = parser.parse_known_args()

    cfg = OmegaConf.load(args.cfg_file)

    if len(sys.argv) > 1 :
        if cfg.dataset.name == 'emdb':
            parts = sys.argv[1].split('_', 1)
            person = parts[0]
            seq = parts[1]
            cfg.dataset.person = person
            cfg.dataset.seq = seq
        elif cfg.dataset.name == 'neuman':
            seq = sys.argv[1]
            cfg.dataset.seq = seq

            noise_power = float(sys.argv[2])
            cfg.noise_power = noise_power
            power_1000 = int(noise_power * 1000)

            suffix = f"{power_1000:03d}_e-3"

            cfg.exp_name = 'noise_power_' + suffix

    else:
        power_1000 = int(cfg.noise_power * 1000)
        suffix = f"{power_1000:03d}_e-3"
        cfg.exp_name = 'noise_power_' + suffix


    get_logger(cfg)
    param_optim(cfg.copy())
    only_train(cfg.copy(), use_refine_data=True)
            