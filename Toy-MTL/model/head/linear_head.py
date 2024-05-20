import torch.nn as nn
import torch

class LinearHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, input_data):
        return self.fc(input_data)
    
class LinearSoftMaxHead(LinearHead):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_data):
        return self.softmax(self.fc(input_data))