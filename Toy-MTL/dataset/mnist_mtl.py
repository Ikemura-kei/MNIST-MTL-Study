import torchvision
import torch

class MNISTMTL(torchvision.datasets.MNIST):
    def __init__(self, cfg, mode, root, transform, download=False):
        super().__init__(root=root, train=mode=='train', transform=transform, download=download)

        self.tasks = None

        if not hasattr(cfg, 'TASK_CFG'):
            raise Exception('No tasks configuration given.')
        
        SUPPORTED_TASKS = [func.replace("TASK_", '') for func in dir(self) if callable(getattr(self, func)) and "TASK_" in func]

        self.tasks = cfg.TASK_CFG.TASKS
        for task in self.tasks:
            if task not in SUPPORTED_TASKS:
                raise Exception('{} is not supported, the list of supported tasks is {}'.format(task, SUPPORTED_TASKS))
            
        self.onehot_enc = torch.eye(10) # 10 classes

    def __getitem__(self, index):
        raw_data = super().__getitem__(index)

        data_dict = {'input_data': raw_data[0]}

        if self.tasks is not None:
            for task in self.tasks:
                data_dict[task] = getattr(self, "TASK_{}".format(task))(raw_data)

        return data_dict
    
    def TASK_CLASSIFICATION_CLE(self, raw_data):
        image, label = raw_data
        
        return label
    
    def TASK_CLASSIFICATION_MAE(self, raw_data):
        image, label = raw_data

        return self.onehot_enc[label]
    
    def TASK_CLASSIFICATION_HUBER(self, raw_data):
        image, label = raw_data

        return self.onehot_enc[label]

    def TASK_NUM_BACKGROUND_PIX_PRED(self, raw_data):
        # -- assume we have normalization --
        std = 0.5
        mean = 0
        orig = raw_data[0] * std + mean
        # print("min {} max {} mean {}".format(torch.min(orig), torch.max(orig), torch.mean(orig)))
        num_background_pix = torch.sum((orig <= 1e-6))
        # print("num_background_pix {}".format(num_background_pix))
        return torch.tensor([num_background_pix.item()])