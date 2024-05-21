import os
import sys

sys.path.append("../UniTR")
from mtl_library.mtl_algo_factory import mtl_algo_factory
from mtl_analysis.mtl_metric_factory import mtl_metric_factory

import torch
import torch.distributed as dist
import numpy as np
from argparse import ArgumentParser
import json
import logging
from datetime import datetime as dt
import datetime
import wandb

from utils.config_utils import (
    load_cfg,
    get_model,
    get_losses,
    get_optim_and_sched,
    get_eval_meter,
    get_dataset,
)
from utils.train_utils import train_one_epoch
from utils.eval_utils import validation


def set_seed(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=datetime.timedelta(0, 3600 * 2)
    )

    parser = ArgumentParser()
    parser.add_argument(
        "--cfg_file",
        type=str,
        required=True,
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Mode of execution, if train is chosen, testing will also be performed at the end.",
    )
    parser.add_argument(
        "--ckpt", type=str, default="", help="Path to the checkpoint to load from."
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="Node rank for distributed training, can be passed from PyTorch distributed launcher.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="The parent folder to contain all experiment logs.",
    )

    # -- load configurations --
    args = parser.parse_args()
    cfg = load_cfg(args)

    # -- set seed --
    set_seed(cfg.SEED)

    # -- create logging directory --
    data_time = dt.now().strftime("%Y-%m-%d_%H:%M:%S")
    # -- we set the output directory to be "<log_dir>/<cfg>/<date_time>" where <cfg> is the name of the config file without .yaml extension --
    output_dir = os.path.join(
        args.log_dir, args.cfg_file.split("/")[-1].replace(".yaml", ""), data_time
    )
    os.makedirs(output_dir, exist_ok=True)

    # -- create logger --
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    log_file = os.path.join(output_dir, "log.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logger.info("=== Configurations ===")
    logger.info(json.dumps(cfg, indent=4))

    logger.info("local rank: %s" % args.local_rank)
    torch.cuda.set_device(args.local_rank)
    torch.backends.cudnn.benchmark = True

    # -- create model --
    model = get_model(cfg).cuda()
    logger.info(model)

    # -- create MTL algorithm --
    shared_weights = [
        param
        for name, param in model.backbone.named_parameters()
        if "weight" in name and "bn" not in name
    ]
    mtl_algo = mtl_algo_factory(
        cfg, next(model.parameters()).device, shared_weights, cfg.TASK_CFG.TASKS
    )
    logger.info("Num shared parameters, {}".format(len(shared_weights)))

    # -- create MTL metric --
    mtl_metrics = mtl_metric_factory(cfg)
    logger.info(mtl_metrics)

    # -- make model distributed --
    model.train()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=False,
    )

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

    # -- create dataset and dataloaders --
    dataset = get_dataset(cfg, "train")
    test_data = get_dataset(cfg, "test")
    train_data, val_data = torch.utils.data.random_split(dataset, [50000, 10000])

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data, drop_last=True
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=cfg.DATASET_CFG.BATCH_SIZE,
        num_workers=4,
        sampler=train_sampler,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_data,
        batch_size=100,
        shuffle=False,
        num_workers=4,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=100,
        shuffle=False,
        num_workers=4,
    )

    start_epoch = 0
    accumulated_iter = 0
    # -- load checkpoint if needed --

    # -- initialize wandb --
    wandb.login()
    run = wandb.init(
        entity="ikemura_kei",
        project="TOY_MTL",
        name=args.cfg_file.split("/")[-1].replace(".yaml", "") + "_" + data_time,
        dir=output_dir,
        mode="online",
        config=cfg,
    )

    # -- start training --
    mtl_dict = {}
    required_keys = [
        "avg_losses",
        "initial_gradient",
        "initial_gradient_norm",
        "ori_gradients",
        "new_gradients",
        "losses",
        "updated_losses",
    ]
    for key in required_keys:
        mtl_dict[key] = None

    for epoch in range(start_epoch, cfg.TRAIN_CFG.EPOCH):
        train_sampler.set_epoch(epoch)

        logger.info(
            "Epoch: {}/{}, lr={}".format(
                epoch, cfg.TRAIN_CFG.EPOCH, optimizer.param_groups[0]["lr"]
            )
        )

        accumulated_iter, result_dict, mtl_dict = train_one_epoch(
            model,
            optimizer,
            losses,
            train_dataloader,
            accumulated_iter,
            mtl_algo,
            epoch,
            mtl_dict,
            cfg,
            args.local_rank,
            mtl_metrics,
        )

        scheduler.step()

        if (
            args.local_rank == 0
            and epoch % cfg.TRAIN_CFG.EVAL_FREQ == 0
            or epoch == (cfg.TRAIN_CFG.EPOCH - 1)
        ):
            result_dict = validation(model, val_dataloader, eval_meter)
            logger.info("Validation: {}".format(result_dict))
            
            log_dict = {}
            for k, v in result_dict.items():
                for i, met_v in enumerate(v):
                    met_name = type(eval_meter[k][i]).__name__
                    log_dict["val/{}/{}".format(k, met_name)] = float(met_v)
                    
            wandb.log({'epoch': epoch, 'iteration': accumulated_iter, **log_dict})

if __name__ == "__main__":
    main()
