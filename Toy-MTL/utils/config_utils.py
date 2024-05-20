from easydict import EasyDict
import yaml
import os

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