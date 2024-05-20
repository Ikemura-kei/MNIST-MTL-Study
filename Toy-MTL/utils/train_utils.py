import torch
from tqdm import tqdm
from torch.utils.data import DataLoader


def train_one_epoch(
    model: torch.nn.Module,
    optimizer,
    loss_fn: torch.nn.ModuleDict,
    train_loader: DataLoader,
    accumulated_iter: int,
    mtl_algo,
    epoch: int,
    mtl_dict: dict,
):
    result_dict = {}

    pbar = tqdm(train_loader)
    avg_losses = {}
    cnt = 0
    
    for sample in pbar:
        cnt += 1
        model.train()
        
        for k, v in sample.items():
            sample[k] = v.cuda()

        optimizer.zero_grad()

        # -- forward pass --
        out = model(sample)

        # -- compute losses --
        losses = {}
        for task_name, fn in loss_fn.items():
            losses[task_name] = fn(out[task_name], sample[task_name])

        # -- perform MTL algorithm --
        mtl_dict = mtl_algo(losses, mtl_dict=mtl_dict, iteration=accumulated_iter)

        # -- update model parameters --
        optimizer.step()

        accumulated_iter += 1

        losses_ = {}
        for task, v in losses.items():
            losses_[task] = "{:3f}".format(v.item())
            avg_losses[task] = v.item() if cnt == 1 else (v.item() + avg_losses[task] * (cnt-1)) / cnt
            
        pbar.set_description("Train losses {}".format(losses_))

    avg_losses_ = {}
    for task, v in avg_losses.items():
        avg_losses_[task] = "{:3f}".format(v)
    print("Average train losses {}".format(avg_losses_))
    
    return accumulated_iter, result_dict, mtl_dict
