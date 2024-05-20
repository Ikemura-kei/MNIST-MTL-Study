import sys
import os

import torch
from argparse import ArgumentParser
import json
import logging
import time
from datetime import datetime

from utils.config_utils import load_cfg

def main():
    parser = ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True, help='Path to the configuration YAML file.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], \
                        help='Mode of execution, if train is chosen, testing will also be performed at the end.')
    parser.add_argument('--ckpt', type=str, default='', help='Path to the checkpoint to load from.')
    parser.add_argument('--local_rank', type=int, default=0, help='Node rank for distributed training, can be passed from PyTorch distributed launcher.')
    parser.add_argument('--log_dir', type=str, default='./logs', help='The parent folder to contain all experiment logs.')

    args = parser.parse_args()
    cfg = load_cfg(args)

    data_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    output_dir = os.path.join(args.log_dir, args.cfg_file.split('/')[-1].replace('.yaml', ''), data_time)
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    log_file = os.path.join(output_dir, 'log.txt')
    logging.basicConfig(filename=log_file, level=logging.INFO)

    logger.info(json.dumps(cfg, indent=4))

if __name__ == "__main__":
    main()