import torch
from tqdm import tqdm

def validation(model, val_loader, eval_meter):
    avg_metrics = {}
    cnt = 0
    
    model.eval()
    for sample in tqdm(val_loader):
        cnt += 1
        
        for k, v in sample.items():
            sample[k] = v.cuda()
            
        with torch.no_grad():
            out = model(sample)
            
        for task, meter in eval_meter.items():
            if cnt == 1:
                avg_metrics[task] = [0] * len(meter)
                
            for i, metric in enumerate(meter):
                avg_metrics[task][i] = metric(out[task], sample[task]) if cnt==1 else (metric(out[task], sample[task]) + avg_metrics[task][i] * (cnt-1)) / cnt
    
    avg_metrics_ = {}
    for k, v in avg_metrics.items():
        avg_metrics_[k] = []
        for v_ in v:
            avg_metrics_[k].append("{:3f}".format(v_.item()))
        
    return avg_metrics_