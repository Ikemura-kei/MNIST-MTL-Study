import torch
import torch.nn as nn

class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute accuracy

        Args:
            pred (torch.Tensor): tensor of shape (N, C), where C is the number of classes, typically is the class probabilities or logits
            gt (torch.Tensor): tensor of shape (N, ), each is an integer representing the class index

        Returns:
            float: accuracy
        """
        N = gt.shape[0]

        pred_label = torch.argmax(pred, dim=1) # (N,)
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
