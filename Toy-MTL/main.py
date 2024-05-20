import sys
import os

import torch
import torch.distributed as dist
import numpy as np
from argparse import ArgumentParser
import json
import logging
import time
from datetime import datetime as dt
import datetime

from utils.config_utils import load_cfg, get_model, get_losses, get_optim_and_sched, get_eval_meter

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(0, 3600*2))

    parser = ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True, help='Path to the configuration YAML file.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], \
                        help='Mode of execution, if train is chosen, testing will also be performed at the end.')
    parser.add_argument('--ckpt', type=str, default='', help='Path to the checkpoint to load from.')
    parser.add_argument('--local-rank', type=int, default=0, help='Node rank for distributed training, can be passed from PyTorch distributed launcher.')
    parser.add_argument('--log_dir', type=str, default='./logs', help='The parent folder to contain all experiment logs.')

    # -- load configurations --
    args = parser.parse_args()
    cfg = load_cfg(args)

    # -- set seed --
    set_seed(cfg.SEED)

    # -- create logging directory --
    data_time = dt.now().strftime("%Y-%m-%d_%H:%M:%S")
    # -- we set the output directory to be "<log_dir>/<cfg>/<date_time>" where <cfg> is the name of the config file without .yaml extension --
    output_dir = os.path.join(args.log_dir, args.cfg_file.split('/')[-1].replace('.yaml', ''), data_time)
    os.makedirs(output_dir, exist_ok=True)

    # -- create logger --
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    log_file = os.path.join(output_dir, 'log.txt')
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logger.info("=== Configurations ===")
    logger.info(json.dumps(cfg, indent=4))

    logger.info('local rank: %s' %args.local_rank)
    torch.cuda.set_device(args.local_rank)
    torch.backends.cudnn.benchmark = True

    # -- create model --
    model = get_model(cfg).cuda()
    logger.info(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], \
                                                      output_device=args.local_rank, find_unused_parameters=False)
    
    # -- create loss functions --
    losses = get_losses(cfg).cuda()
    logger.info(losses)

    # -- create optimizer and scheduler --
    optimizer, scheduler = get_optim_and_sched(cfg, model)
    logger.info(optimizer)
    logger.info(scheduler)

    # -- create evaluation meters --
    eval_meter = get_eval_meter(cfg)
    logger.info(eval_meter)

if __name__ == "__main__":
    main()