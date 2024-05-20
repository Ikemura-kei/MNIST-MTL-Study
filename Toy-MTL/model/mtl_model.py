import torch.nn as nn
import torch

class MTLModel(nn.Module):
    def __init__(self, backbone: nn.Module, heads: nn.ModuleDict):
        super().__init__()
        self.backbone = backbone
        self.heads = heads

    def forward(self, input_dict: dict) -> dict:
        """Forward compute

        Args:
            input_dict (dict): the input dictionary containing at least the following keys:
                               - input_data: the input image data

        Returns:
            dict: the output dictionary, contains the following key-value pairs
                  - shared_feat: the shared feature
                  - <task_name>: task-wise output
        """
        output_dict = {}

        input_data = input_dict['input_data']
        shared_feat = self.backbone(input_data)

        output_dict['shared_feat'] = shared_feat

        for task, head in self.heads.items():
            output_dict[task] = head(shared_feat)

        return output_dict
        