from easydict import EasyDict
import yaml
import os
import torch.nn as nn
import model.mtl_model
import model
import loss

def load_cfg(args) -> EasyDict:
    """Load configuration as an EasyDict object

    Args:
        args (argparse.Namespace): the parsed arguments stored as a Namespace object.

    Returns:
        EasyDict: the resulting full configuration.
    """
    
    cfg = EasyDict()
    
    # -- read yaml file and copy variables --
    assert os.path.exists(args.cfg_file)
    with open(args.cfg_file) as stream:
        raw_cfg = yaml.safe_load(stream)
    cfg = recursive_dict_to_easydict(raw_cfg, cfg)

    return cfg

def recursive_dict_to_easydict(dictionary: dict, easydict: EasyDict) -> EasyDict:
    """Convert a dictionary to EasyDict recursively if the dictionary contains another dictionary

    Args:
        dictionary (dict): the input dictionary
        easydict (EasyDict): the input easydict

    Returns:
        EasyDict: the output EasyDict object
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):
            easydict[key] = EasyDict()
            easydict[key] = recursive_dict_to_easydict(value, easydict[key])
        else:
            easydict[key] = value

    return easydict

def get_model(cfg: EasyDict) -> nn.Module:
    backbone = getattr(model, "{}Backbone".format(cfg.MODEL_CFG.BACKBONE_CFG.TYPE))(**cfg.MODEL_CFG.BACKBONE_CFG.PARAMS)

    heads = nn.ModuleDict()
    for task in cfg.TASK_CFG.TASKS:
        head_cfg = getattr(cfg.MODEL_CFG.HEADS_CFG, task)
        head = getattr(model, "{}Backbone".format(head_cfg.TYPE))(**head_cfg.PARAMS)
        heads[task] = head

    net = model.mtl_model.MTLModel(backbone, heads)
    return net

def get_losses(cfg: EasyDict) -> nn.ModuleDict:
    task_cfg = cfg.TASK_CFG

    losses = nn.ModuleDict()
    for task in task_cfg.TASKS:
        loss_fn = getattr(loss, getattr(task_cfg, task).LOSS)(**getattr(task_cfg, task).PARAMS)
        losses[task] = loss_fn
    
    return losses