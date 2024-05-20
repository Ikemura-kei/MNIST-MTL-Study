import torch
import torch.nn as nn

class Accuracy(nn.Module):
    def __init__(self, gt_is_logit=False):
        super().__init__()
        self.gt_is_logit = gt_is_logit

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute accuracy

        Args:
            pred (torch.Tensor): tensor of shape (N, C), where C is the number of classes, typically is the class probabilities or logits
            gt (torch.Tensor): 
                self.gt_is_logit=False: tensor of shape (N, ), each is an integer representing the class index
                self.gt_is_logit=True: tensor of shape (N, C), each is an onehot encoding of the class label

        Returns:
            float: accuracy
        """
        N = gt.shape[0]

        pred_label = torch.argmax(pred, dim=1) # (N,)
        
        if self.gt_is_logit:
            gt = torch.argmax(gt, dim=1) # (N,)
            
        acc = torch.sum(torch.eq(pred_label, gt)) / N
        
        return acc

if __name__ == "__main__":
    gt = torch.Tensor([1, 2, 3, 0, 2, 4])
    pred = torch.rand((6, 5))
    pred[0][1] = 1e10 # correct
    pred[1][0] = 1e10 # wrong
    pred[2][3] = 1e10 # correct
    pred[3][0] = 1e10 # correct
    pred[4][2] = 1e10 # correct
    pred[5][2] = 1e10 # wrong

    acc_meter = Accuracy()

    print(acc_meter(pred, gt))
