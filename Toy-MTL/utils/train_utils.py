import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb

def train_one_epoch(
    model: torch.nn.Module,
    optimizer,
    loss_fn: torch.nn.ModuleDict,
    train_loader: DataLoader,
    accumulated_iter: int,
    mtl_algo,
    epoch: int,
    mtl_dict: dict,
    cfg,
    local_rank,
    mtl_metrics,
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
            
        # # -- initialize MTL metric data if needed --
        # if local_rank == 0:
        #     if mtl_dict[list(mtl_dict.keys())[0]] is None:
        #         mtl_dict['losses'] = {}
        #         mtl_dict['updated_losses'] = {}
        #         mtl_dict['avg_losses'] = {}
        #         mtl_dict['ori_gradients'] = {}
        #         mtl_dict['new_gradients'] = {}
        #         mtl_dict['initial_gradient_norm'] = {}
        #         mtl_dict['initial_gradient'] = {}
        #         for i, key in enumerate(list(losses.keys())):
        #             mtl_dict['losses'][key] = None
        #             mtl_dict['updated_losses'][key] = None
        #             mtl_dict['avg_losses'][key] = None
        #             mtl_dict['initial_gradient_norm'][key] = None
        #             mtl_dict['initial_gradient'][key] = None

        # -- perform MTL algorithm --
        mtl_dict = mtl_algo(losses, mtl_dict=mtl_dict, iteration=accumulated_iter)

        # -- update model parameters --
        optimizer.step()
        
        # -- record MTL metric data --
        if local_rank == 0 and len(mtl_metrics) > 0:
            mtl_dict['ori_gradients'] = mtl_algo.grads
            
            init_losses = mtl_dict['losses'] is None
            if init_losses:
                mtl_dict['losses'] = {}
            
            init_init_grad = mtl_dict['initial_gradient_norm'] is None
            if init_init_grad:
                mtl_dict['initial_gradient_norm'] = {}
                mtl_dict['initial_gradient'] = {}
                
            init_avg_grad = mtl_dict['avg_losses'] is None
            if init_avg_grad:
                mtl_dict['avg_losses'] = {}
            
            for j, key in enumerate(list(losses.keys())):
                if init_losses:
                    mtl_dict['losses'][key] = torch.tensor([losses[key].item()])
                else:
                    mtl_dict['losses'][key] = torch.cat([mtl_dict['losses'][key], torch.tensor([losses[key].item()])], dim=0)

                if init_init_grad:
                    mtl_dict['initial_gradient'][key] = mtl_algo.grads[key]
                    mtl_dict['initial_gradient_norm'][key] = torch.norm(mtl_algo.grads[key])

                if init_avg_grad:
                    mtl_dict['avg_losses'][key] = torch.tensor([losses[key].item()])
                elif (accumulated_iter+1) % cfg.MTL_CONFIG.ANA.LOSS_AVERAGING_WINDO_SIZE == 0:
                    idx_start = accumulated_iter - (cfg.MTL_CONFIG.ANA.LOSS_AVERAGING_WINDO_SIZE-1)
                    idx_end = accumulated_iter+1
                    avg = torch.mean(mtl_dict['losses'][key][idx_start:idx_end])
                    mtl_dict['avg_losses'][key] = torch.cat([mtl_dict['avg_losses'][key], torch.tensor([avg.item()])], dim=0)

            # -- log mtl metrics to wandb --
            mtl_metric_log_dict = {}
            for metric in mtl_metrics:
                mtl_dict = metric(mtl_dict)
                mtl_metric_log_dict = metric.log_to_dict(mtl_metric_log_dict, stage='train')
            wandb.log({'iteration': accumulated_iter, **mtl_metric_log_dict})

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
