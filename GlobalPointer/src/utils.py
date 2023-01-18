import random

import numpy as np
import torch


def move_dict_value_to_device(*args, device):
    for arg in args:
        for key, value in arg.items():
            if isinstance(value, torch.Tensor):
                arg[key] = value.to(device)


def collate_fn(batch):
    packed_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            packed_batch[key] = torch.stack([s[key] for s in batch])
        else:
            packed_batch[key] = [s[key] for s in batch]

    return packed_batch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
