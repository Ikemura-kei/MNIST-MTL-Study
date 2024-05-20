import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import argparse
import matplotlib.pyplot as plt
import numpy as np
import random 

from utils import projection2simplex, find_min_norm_element, set_seed
from moo import grad_ew, grad_mgda, grad_moco, grad_modo, grad_fixed_weights

import os
import sys
sys.path.append(os.path.abspath('../UniTR'))
from mtl_analysis.mtl_metric_factory import mtl_metric_factory

import yaml
metric_cfg = './config.yaml'
assert os.path.exists(metric_cfg)
from easydict import EasyDict

import wandb
import warnings

old = False

def load_cfg(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        load_cfg(config[key], val)

    return config

with open(metric_cfg) as stream:
    cfg = yaml.safe_load(stream)
cfg = load_cfg(EasyDict(), cfg)
mtl_metrics = mtl_metric_factory(cfg)
wandb.login()
    
# Create arg parser

parser = argparse.ArgumentParser(description='Arguments for Toy MOO task')

# general
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_epochs', type=int, default=4250, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate for the model')
parser.add_argument('--moo_method', type=str, default='FixedWeights', help='MOO method for updating model, option: EW, MGDA, MoCo, MoDo')

# moo method specific
# EW
# None
# MGDA
# None
# MoCo
parser.add_argument('--beta_moco', type=float, default=0.1, help='learning rate of tracking variable')
parser.add_argument('--gamma_moco', type=float, default=0.1, help='learning rate of lambda')
parser.add_argument('--rho_moco', type=float, default=0.0, help='regularization parameter of lambda subproblem')
# MoDo
parser.add_argument('--gamma_modo', type=float, default=0.1, help='learning rate of lambda')
parser.add_argument('--rho_modo', type=float, default=0.0, help='regularization parameter')

# parse args
params = parser.parse_args()
print(params)

# General hyper-parameters

# seed
seed = params.seed
# batch size for sampling data
batch_size = params.batch_size
# training iterations
num_epochs = params.num_epochs
# learning rate of model
lr = params.lr
# MOO method
moo_method = params.moo_method

# MOO method specific hyper-parameters

# EW
# None

# MoCo
moco_beta = params.beta_moco
moco_gamma = params.gamma_moco
moco_rho = params.rho_moco

# MoDo
modo_gamma = params.gamma_modo
modo_rho = params.rho_modo
weights_candidate = [[0.1, 0.1, 0.8], [0.1, 0.2, 0.7], [0.1, 0.3, 0.6], [0.1, 0.4, 0.5], [0.1, 0.5, 0.4], [0.1, 0.6, 0.3], [0.1, 0.7, 0.2], [0.1, 0.8, 0.1],
           [0.2, 0.1, 0.7], [0.3, 0.1, 0.6], [0.4, 0.1, 0.5], [0.5, 0.1, 0.4], [0.6, 0.1, 0.3], [0.7, 0.1, 0.2], [0.8, 0.1, 0.1], [0.7, 0.2, 0.1],
           [0.6, 0.3, 0.1], [0.5, 0.4, 0.1], [0.4, 0.5, 0.1], [0.3, 0.6, 0.1], [0.2, 0.7, 0.1]]
weights = weights_candidate[14]
run = wandb.init(
    entity = 'ikemura_kei',
    project='MNIST_MTL',
    name='w1_{:1f}-w2_{:1f}-w3_{:1f}'.format(weights[0], weights[1], weights[2]),
    dir='logs/debug/output',
    mode='online',
    config=cfg,
)
# calc accuracy of predictions
def get_accuracy(pred, label):
    return torch.sum(torch.eq( torch.argmax(pred, dim=1), label )) / pred.shape[0]

SUPPORTED_TASKS = ['CLASSIFICATION', 'CLASSIFICATION2', 'CLASSIFICATION3', 'NUM_BACKGROUND_PIX_PRED']
class MNISTMTL(torchvision.datasets.MNIST):
    def __init__(self, root, train, transform, download, cfg):
        super().__init__(root=root, train=train, transform=transform, download=download)

        self.tasks = None

        if not hasattr(cfg, 'TASKS'):
            raise Exception('No tasks configuration given.')
            return

        self.tasks = cfg.TASKS
        for task in self.tasks:
            if task not in SUPPORTED_TASKS:
                raise Exception('{} is not supported, the list of supported tasks is {}'.format(task, SUPPORTED_TASKS))

    def __getitem__(self, index):
        raw_data = super().__getitem__(index)

        data_dict = {'image': raw_data[0]}

        if self.tasks is not None:
            for task in self.tasks:
                data_dict[task] = getattr(self, task)(raw_data)

        return data_dict
    
    def CLASSIFICATION(self, raw_data):
        image, label = raw_data

        return label
    
    def CLASSIFICATION2(self, raw_data):
        image, label = raw_data

        return label
    
    def CLASSIFICATION3(self, raw_data):
        image, label = raw_data

        return label

    def NUM_BACKGROUND_PIX_PRED(self, raw_data):
        # -- assume we have normalization --
        std = 0.5
        mean = 0
        orig = raw_data[0] * std + mean
        # print("min {} max {} mean {}".format(torch.min(orig), torch.max(orig), torch.mean(orig)))
        num_background_pix = torch.sum((orig <= 1e-6))
        # print("num_background_pix {}".format(num_background_pix))
        return torch.tensor([num_background_pix.item()])

# get performance measures of the learned model
def get_performance_old(model, optimizer, dataloader, loss_dict, num_param, num_param_layer, softmax, onehot_enc):
    grad_list = 0
    loss_list = 0
    acc = 0
    count = 0
    for data, label in iter(dataloader):
        data = data.cuda()
        label = label.cuda()
        pred = model(data)
        grad_list_, loss_list_ = get_grads_old(model, optimizer, pred, label, loss_dict, num_param, num_param_layer, softmax, onehot_enc)
        grad_list += grad_list_
        loss_list += loss_list_
        acc += get_accuracy(pred.detach(), label)

        count += 1
    
    grad_list /= count
    loss_list /= count

    lambd_opt = find_min_norm_element(grad_list)
    multi_grad = lambd_opt @ grad_list
    
    return acc.item()/count, loss_list, torch.norm(multi_grad).item()

def get_performance(model, optimizer, dataloader, loss_fn_dict, num_param, num_param_layer, softmax, onehot_enc):
    model.eval()
    grad_list = 0
    loss_list = 0
    acc = {'CLASSIFICATION': 0, 'CLASSIFICATION2': 0, 'CLASSIFICATION3': 0}
    count = 0
    for data in iter(dataloader):
        for k, v in data.items():
            data[k] = v.cuda()

        pred = model(data)
        grad_list_, loss_list_, grad_dict, loss_dict = get_grads(model, optimizer, pred, data, loss_fn_dict, num_param, num_param_layer, softmax, onehot_enc, num_losses)
        grad_list += grad_list_
        loss_list += loss_list_
        for key in ['CLASSIFICATION', 'CLASSIFICATION2', 'CLASSIFICATION3']:
            acc[key] += get_accuracy(pred[key].detach(), data[key])

        count += 1
    
    grad_list /= count
    loss_list /= count

    lambd_opt = find_min_norm_element(grad_list)
    multi_grad = lambd_opt @ grad_list

    for key in ['CLASSIFICATION', 'CLASSIFICATION2', 'CLASSIFICATION3']:
         acc[key] = acc[key].item()/count
    
    return acc, loss_list, torch.norm(multi_grad).item()

# get layer-wise parameter numbers
def get_layer_params_old(model):
    # init layer-wise param number list
    num_param_layer = []

    # print layer-wise parameter numbers
    print("\n"+"="*50)
    print('Model parameter count per layer')
    print("="*50)
    # get layerwise param numbers, with layer names
    for name, param in model.named_parameters():
        num_param_layer.append(param.data.numel())
        print(f'{name}', f'\t: {param.data.numel()}')
    print('Total number of parametrs :', sum(num_param_layer))
    print("-"*50)
    # return layerwise and total param numbers
    return sum(num_param_layer), num_param_layer


def get_layer_params(model):
    # init layer-wise param number list
    num_param_layer = []

    # print layer-wise parameter numbers
    print("\n"+"="*50)
    print('Model parameter count per layer')
    print("="*50)
    # get layerwise param numbers, with layer names
    for name, param in model.get_shared_params():
        num_param_layer.append(param.data.numel())
        print(f'{name}', f'\t: {param.data.numel()}')
    print('Total number of parametrs :', sum(num_param_layer))
    print("-"*50)
    # return layerwise and total param numbers
    return sum(num_param_layer), num_param_layer

# get vectorized grad information
def get_grad_vec_old(model, num_param, num_param_layer):
    # initialize grad with a vecotr with size num. param.
    grad_vec = torch.zeros(num_param)
    # count params to put grad blocks in correct index of vector
    count = 0
    for param in model.parameters():
        # collect grad only if not None, else return zero vector
        if param.grad is not None:
            # calculate vecotr block start and end indices
            beg = 0 if count == 0 else sum(num_param_layer[:count])
            end = sum(num_param_layer[:(count+1)])
            # put flattened grad param into the vector block
            grad_vec[beg:end] = param.grad.data.view(-1)
        count += 1
    
    return grad_vec

def get_grad_vec(model, num_param, num_param_layer):
    # initialize grad with a vecotr with size num. param.
    grad_vec = torch.zeros(num_param)
    # count params to put grad blocks in correct index of vector
    count = 0
    for name, param in model.get_shared_params():
        # collect grad only if not None, else return zero vector
        if param.grad is not None:
            # calculate vecotr block start and end indices
            beg = 0 if count == 0 else sum(num_param_layer[:count])
            end = sum(num_param_layer[:(count+1)])
            # put flattened grad param into the vector block
            grad_vec[beg:end] = param.grad.data.view(-1)
        count += 1
    
    return grad_vec

# get gradient and loss values w.r.t each loss function
def get_grads_old(model, optimizer, pred, label, loss_dict, num_param, num_param_layer, softmax, onehot_enc):
    # init gradient list (to be collected one gradient for each loss)
    grad_list = []
    loss_list = []
    # to switch off retain_graph in loss.backward()
    num_loss = len(loss_dict) 
    # compute the loss w.r.t each loss function
    for k, loss_fn in enumerate(loss_dict):
        # print(loss_fn)
        # print(pred.shape)
        # print(label)
        # print(loss_dict[loss_fn])
        if loss_fn =='mse' or loss_fn =='huber':
            loss = loss_dict[loss_fn](softmax(pred), onehot_enc[label])
        else:
            loss = loss_dict[loss_fn](pred, label)
        # make gradient of model zero
        optimizer.zero_grad()
        # compute loss w.r.t current loss function
        loss.backward(retain_graph=True) if k < num_loss - 1 else loss.backward()
        # compute vectorized gradient
        grad_vec = get_grad_vec_old(model, num_param, num_param_layer)
        # collect the gradient for current loss
        grad_list.append(grad_vec)
        loss_list.append(loss.detach().item())
    
    return torch.stack(grad_list), np.array(loss_list)

def get_grads(model, optimizer, pred, label, loss_fn_dict, num_param, num_param_layer, softmax, onehot_enc, num_loss):
    # pred: a dictionary with task names as keys, and each is the prediction of the corresponding task
    # label: a dictionary with task names as keys, and each is the ground truth value(s) of the corresponding task
    # loss_fn_dict: a dictionary with task names as keys, and each element is another dictionary of metrics (loss functions) for that task
    # num_loss: the total number of loss functions to compute, this is used to know when we reach the last loss and then we can turn off retain_graph in backward

    # init gradient list (to be collected one gradient for each loss)
    grad_dict = {}
    loss_dict = {}
    grad_list = []
    loss_list = []
    # -- initialize returned dictionaries --
    for task in list(loss_fn_dict.keys()):
        grad_dict[task] = {}
        loss_dict[task] = {}
        for k, loss_fn in enumerate(loss_fn_dict[task]):
            grad_dict[task][loss_fn] = None
            loss_dict[task][loss_fn] = None

    # compute the loss w.r.t each loss function
    for task in list(loss_fn_dict.keys()):
        for k, loss_fn in enumerate(loss_fn_dict[task]):
            # print('loss {}'.format(loss_fn))
            # print('pred {} {}'.format(pred[task][:2], softmax(pred[task][:2])))
            # print('label {}'.format(label[task][:2]))
            # print(loss_fn_dict[task][loss_fn])
            if loss_fn =='MSELoss' or loss_fn =='HuberLoss':
                loss = loss_fn_dict[task][loss_fn](softmax(pred[task]), onehot_enc[label[task]])
            else:
                loss = loss_fn_dict[task][loss_fn](pred[task], label[task])
                
            # -- reset model gradients --
            for name, param in model.get_shared_params():
                # put grad only if grad is initialized
                if param.grad is not None:
                    param.grad = torch.zeros_like(param.data)

            # -- compute gradient w.r.t current loss function -- 
            loss.backward(retain_graph=True) if k < num_loss - 1 else loss.backward()

            # -- compute vectorized gradient --
            grad_vec = get_grad_vec(model, num_param, num_param_layer)

            # -- collect the gradient for current loss --
            grad_dict[task][loss_fn] = (grad_vec)
            grad_list.append(grad_vec)
            loss_dict[task][loss_fn] = loss.detach().item()
            loss_list.append(loss_dict[task][loss_fn])
    
    return torch.stack(grad_list), np.array(loss_list), grad_dict, loss_dict

# set multi-gradient in the model param grad
def set_grads_old(model, multi_grad, num_param_layer):
    # count params to put multi-grad blocks in correct model param grad
    count = 0
    for param in model.parameters():
        # put grad only if grad is initialized
        if param.grad is not None:
            # calculate vector block start and end indices
            beg = 0 if count == 0 else sum(num_param_layer[:count])
            end = sum(num_param_layer[:(count+1)])
            # put reshaped multi-grad block into model param grad
            param.grad.data = multi_grad[beg:end].view(param.data.size()).data.clone()
        count += 1   
    return 

def set_grads(model, multi_grad, num_param_layer):
    # count params to put multi-grad blocks in correct model param grad
    count = 0
    for name, param in model.get_shared_params():
        # put grad only if grad is initialized
        if param.grad is not None:
            # calculate vector block start and end indices
            beg = 0 if count == 0 else sum(num_param_layer[:count])
            end = sum(num_param_layer[:(count+1)])
            # put reshaped multi-grad block into model param grad
            param.grad.data = multi_grad[beg:end].view(param.data.size()).data.clone()
        count += 1   
    return 

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.input_dim = 28 * 28 # input image data in vector form
        self.hidden_dim = 512 # hidden layer size
        self.output_dim = 10 # number of digit classes

        # define model layers
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, inputs):
        x = inputs.view(inputs.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x
    
class MTLModel(nn.Module):
    def __init__(self, cfg):
        super(MTLModel, self).__init__()
        # -- use fixed input and hidden dimension, though can be made configurable later --
        self.input_dim = 28 * 28 # input image data in vector form
        self.hidden_dim = 512 # hidden layer size
        self.tasks = None

        # -- define shared layers --
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)

        # -- define task heads --
        if not hasattr(cfg, 'TASKS'):
            raise Exception('No tasks configuration given, will only perform digit classification on MNIST.')

        self.tasks = cfg.TASKS
        for task in self.tasks:
            if task not in SUPPORTED_TASKS:
                raise Exception('{} is not supported, the list of supported tasks is {}'.format(task, SUPPORTED_TASKS))

        for task in self.tasks:
            output_dim = getattr(cfg, task).OUTPUT_DIM
            setattr(self, "task_head_{}".format(task), nn.Linear(self.hidden_dim, output_dim))
        
    def forward(self, input_dict):
        image = input_dict['image']
        x = image.view(image.shape[0], -1)
        shared_feat = self.fc1(x)
        
        ret_dict = {}

        if self.tasks is not None:
            for task in self.tasks:
                ret_dict[task] = getattr(self, "task_head_{}".format(task))(shared_feat)

        return ret_dict
    
    def get_shared_params(self):
        ret = []
        for param in self.named_parameters():
            name = param[0]
            if 'task_head' not in name:
                ret.append(param)

        return ret

# Create toy datset and the data_loaders

# Define the transforms for data preprocessing
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.,), (0.5,))]
)

# set seed
seed_worker, g = set_seed(seed)

# Load the MNIST dataset 

# half batch size if MoDo
if moo_method == 'MoDo':
    batch_size = batch_size//2
# get the initial training dataset
if old:
    dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
else:
    dataset = MNISTMTL(root='./data', train=True, transform=transform, download=False, cfg=cfg)
# split the data set into train and val
train_data, val_data = torch.utils.data.random_split(dataset, [50000, 10000])
# creare train loader for training
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, worker_init_fn=seed_worker, generator=g)
# create another train loader for evaluation
train_eval_dataloader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=False, num_workers=2, worker_init_fn=seed_worker, generator=g)
# create val loader
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=100, shuffle=False, num_workers=2, worker_init_fn=seed_worker, generator=g)
# get test dataset and create test dataloader
if old:
    test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
else:
    test_data = MNISTMTL(root='./data', train=False, transform=transform, download=False, cfg=cfg)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False, num_workers=2, worker_init_fn=seed_worker, generator=g)


# Set-up for training

# init model
if old:
    model = Model()
    model = model.cuda()
else:
    model = MTLModel(cfg)
    model = model.cuda()
    for name, param in model.get_shared_params():
        print(name)

# get layerwise parameter numbers
if old:
    num_param, num_param_layer = get_layer_params_old(model)
else:
    num_param, num_param_layer = get_layer_params(model)

# Defining loss functions 

# onehot encoding for classes
onehot_enc = torch.eye(10)
onehot_enc = onehot_enc.cuda()

# some useful activations
logsoftmax = nn.LogSoftmax(dim=1)
softmax = nn.Softmax(dim=1)

# Loss functions

# cross-netropy loss (same as nll loss)
cross_entropy_loss = nn.CrossEntropyLoss()
# l1loss
l1_loss = nn.L1Loss()
# hinge loss
hinge_loss = torch.nn.MultiMarginLoss()
# hinge loss
hinge_loss = torch.nn.MultiMarginLoss()
# MSE loss
mse_loss = torch.nn.MSELoss()
# Huber loss
huber_loss = torch.nn.HuberLoss(delta=0.1) # to make sure this is deifferent from mse

# dictionary of losses
if old:
    loss_dict = {'cel':cross_entropy_loss, 'mse':mse_loss, 'huber':huber_loss}
    num_losses = len(loss_dict)
else:
    loss_fn_dict = {}
    num_losses = 0

    for task in cfg.TASKS:
        loss_fn_dict[task] = {}
        task_cfg = getattr(cfg, task)

        for loss_fn in task_cfg.LOSS_FN:
            loss_cfg = getattr(task_cfg, loss_fn)
            if loss_cfg.USE_PYTORCH:
                loss_fn_dict[task][loss_fn] = getattr(torch.nn, loss_fn)(**loss_cfg.PARAMS)
            else:
                raise NotImplementedError
            
            num_losses += 1


# MOO method specific functions and parameters

# collection of multi-grad calculation methods
multi_grad_fn = {'EW': grad_ew, 'MGDA':grad_mgda, 'MoCo': grad_moco, 'MoDo':grad_modo, 'FixedWeights':grad_fixed_weights}

# MOO menthod specific params

# EW
# None

# MGDA
# None

# MoCo
moco_kwargs = {'y':torch.zeros(num_losses, num_param), 'lambd':torch.ones([num_losses, ])/num_losses, 'beta':moco_beta, 'gamma':moco_gamma, 'rho':moco_rho}

# MoDo
modo_kwargs = {'lambd':torch.ones(num_losses)/num_losses, 'gamma':modo_gamma, 'rho':modo_rho}

fixed_weights_kwargs = {'weights': torch.tensor(weights)}

# add kwarg argumnets to one dictionary, so training is general to all methods
kwargs = {'EW':{}, 'MGDA':{}, 'MoCo':moco_kwargs, 'MoDo':modo_kwargs, 'FixedWeights':fixed_weights_kwargs}

# optimiser (sgd)
optimizer = optim.SGD(model.parameters(), lr=lr)

# print log format
print("\n"+"="*70)
print(f'LOG FORMAT: Epoch: EPOCH | Train: {" ".join([f"LOSS{i+1}" for i in range(num_losses)])}')
print("="*70)
mtl_dict = {}
required_keys = ['avg_losses', 'initial_gradient', 'initial_gradient_norm',
                'ori_gradients', 'new_gradients', 'losses', 'updated_losses']

for key in required_keys:
    mtl_dict[key] = None

for i in range(num_epochs):
    print("{}/{}".format(i, num_epochs))
    model.train()
    accumulated_iter = i + 1
    optimizer.zero_grad()

    # if MoDo, double sample
    if moo_method=='MoDo':
        grad_list = []
        loss_list = [0 for _ in range(num_tasks)]
        for j in range(2):
            # sample training data and label
            data, label = next(iter(train_dataloader))  
            # get model prediction (logits)  
            pred = model(data)
            # get gradients and loss values w.r.t each loss fn
            grad_list_, loss_list_ = get_grads(model, optimizer, pred, label, loss_dict, num_param, num_param_layer, softmax, onehot_enc)
            grad_list.append(grad_list_)
            # average the loss over two samples
            loss_list = [loss_list[k] + 0.5 * loss_list_[k] for k in range(num_tasks)]
    # or single batch sample for other methods
    else:
        # sample training data and label
        input_dict = next(iter(train_dataloader))  

        if old:
            for j in range(len(input_dict)):
                input_dict[j] = input_dict[j].cuda()
            pred_dict = model(input_dict[0])
        else:
            for k, v in input_dict.items():
                input_dict[k] = v.cuda()
            pred_dict = model(input_dict)

        # get gradients and loss values w.r.t each loss fn
        if old:
            grad_list, loss_list = get_grads_old(model, optimizer, pred_dict, input_dict[1], loss_dict, num_param, num_param_layer, softmax, onehot_enc)
        else:
            grad_list, loss_list, grad_dict, loss_dict = get_grads(model, optimizer, pred_dict, input_dict, loss_fn_dict, num_param, num_param_layer, softmax, onehot_enc, num_losses)
    
    # -- calc multi-grad according to moo method --
    print(moo_method)
    multi_grad = multi_grad_fn[moo_method](grad_list, **kwargs)
    multi_grad = multi_grad.cuda()

    # -- update model grad with the multi-grad --
    if old:
        set_grads_old(model, multi_grad, num_param_layer)
    else:
        set_grads(model, multi_grad, num_param_layer)

    # -- prepare measurements required by the MTL metrics --
    # --> avg_losses: dict
    # --> initial_gradient: dict
    # --> initial_gradient_norm: dict
    # --> ori_gradients: dict
    # --> new_gradients: dict
    # --> losses: dict
    # --> updated_losses: dict

    # -- initialize mtl_dict with task keys --
    if mtl_dict[list(mtl_dict.keys())[0]] is None:
        mtl_dict['losses'] = {}
        mtl_dict['updated_losses'] = {}
        mtl_dict['avg_losses'] = {}
        mtl_dict['ori_gradients'] = {}
        mtl_dict['new_gradients'] = {}
        mtl_dict['initial_gradient_norm'] = {}
        mtl_dict['initial_gradient'] = {}
        for i, key in enumerate(list(loss_dict.keys())):
            mtl_dict['losses'][key] = None
            mtl_dict['updated_losses'][key] = None
            mtl_dict['avg_losses'][key] = None
            mtl_dict['initial_gradient_norm'][key] = None
            mtl_dict['initial_gradient'][key] = None

    for j, key in enumerate(list(loss_dict.keys())):
        if mtl_dict['losses'][key] is None:
            mtl_dict['losses'][key] = torch.tensor([loss_list[j].item()])
        else:
            mtl_dict['losses'][key] = torch.cat([mtl_dict['losses'][key], torch.tensor([loss_list[j].item()])], dim=0)

        mtl_dict['ori_gradients'][key] = grad_list[j]

        if mtl_dict['initial_gradient_norm'][key] is None:
            mtl_dict['initial_gradient'][key] = grad_list[j]
            mtl_dict['initial_gradient_norm'][key] = torch.norm(grad_list[j])

        if mtl_dict['avg_losses'][key] is None:
            mtl_dict['avg_losses'][key] = torch.tensor([loss_list[j].item()])
        elif (accumulated_iter+1) % cfg.MTL_CONFIG.ANA.LOSS_AVERAGING_WINDO_SIZE == 0:
            idx_start = accumulated_iter - (cfg.MTL_CONFIG.ANA.LOSS_AVERAGING_WINDO_SIZE-1)
            idx_end = accumulated_iter+1
            avg = torch.mean(mtl_dict['losses'][key][idx_start:idx_end])
            mtl_dict['avg_losses'][key] = torch.cat([mtl_dict['avg_losses'][key], torch.tensor([avg.item()])], dim=0)

        # -- TODO: change all that are updated to the actual new values --
        if mtl_dict['updated_losses'][key] is None:
            mtl_dict['updated_losses'][key] = torch.tensor([loss_list[j].item() * weights[j]])
        else:
            mtl_dict['updated_losses'][key] = torch.cat([mtl_dict['updated_losses'][key], torch.tensor([loss_list[j].item() * weights[j]])], dim=0)

        mtl_dict['new_gradients'][key] = grad_list[j] * weights[j]

    # -- log mtl metrics to wandb --
    mtl_metric_log_dict = {}
    for metric in mtl_metrics:
        mtl_dict = metric(mtl_dict)
        mtl_metric_log_dict = metric.log_to_dict(mtl_metric_log_dict, stage='train')
    wandb.log({'epoch': i, **mtl_metric_log_dict})

    for name, param in model.named_parameters():
        norm = torch.norm(param)
        print("Name {}, norm {}".format(name, norm))

    # -- update model param --
    optimizer.step()

    # -- periodic evaluation and saving --
    if i%50 == 0 or i==num_epochs-1:
        print(f"Epoch: {i: 6,} | Train: { ' '.join(str(round(num, 4)) for num in loss_list) }")
        if old:
            train_acc, train_loss, train_ps = get_performance_old(model, optimizer, train_eval_dataloader, loss_dict, num_param, num_param_layer, softmax, onehot_enc)
            test_acc, test_loss, test_ps = get_performance_old(model, optimizer, test_dataloader, loss_dict, num_param, num_param_layer, softmax, onehot_enc)
            val_acc, val_loss, val_ps = get_performance_old(model, optimizer, val_dataloader, loss_dict, num_param, num_param_layer, softmax, onehot_enc)
        else:
            train_acc, train_loss, train_ps = get_performance(model, optimizer, train_eval_dataloader, loss_fn_dict, num_param, num_param_layer, softmax, onehot_enc)
            test_acc, test_loss, test_ps = get_performance(model, optimizer, test_dataloader, loss_fn_dict, num_param, num_param_layer, softmax, onehot_enc)
            val_acc, val_loss, val_ps = get_performance(model, optimizer, val_dataloader, loss_fn_dict, num_param, num_param_layer, softmax, onehot_enc)

        wandb.log({'epoch': i, 'val/loss': val_loss, 'val/ps': val_ps,
       'test/loss': test_loss, 'test/ps': test_ps,
        'train/loss': train_loss, 'train/ps': train_ps})

        acc = {}
        for key in ['CLASSIFICATION', 'CLASSIFICATION2', 'CLASSIFICATION3']:
            acc["{}/train_acc".format(key)] = train_acc[key]
            acc["{}/val_acc".format(key)] = val_acc[key]
            acc["{}/test_acc".format(key)] = test_acc[key]

        wandb.log({'epoch': i, 'test/loss': test_loss, **acc})


# get perforamnce measures from each dataset
if old:
    train_acc, train_loss, train_ps = get_performance_old(model, optimizer, train_eval_dataloader, loss_dict, num_param, num_param_layer, softmax, onehot_enc)
    val_acc, val_loss, val_ps = get_performance_old(model, optimizer, val_dataloader, loss_dict, num_param, num_param_layer, softmax, onehot_enc)
    test_acc, test_loss, test_ps = get_performance_old(model, optimizer, test_dataloader, loss_dict, num_param, num_param_layer, softmax, onehot_enc)
else:
    train_acc, train_loss, train_ps = get_performance(model, optimizer, train_eval_dataloader, loss_fn_dict, num_param, num_param_layer, softmax, onehot_enc)
    val_acc, val_loss, val_ps = get_performance(model, optimizer, val_dataloader, loss_fn_dict, num_param, num_param_layer, softmax, onehot_enc)
    test_acc, test_loss, test_ps = get_performance(model, optimizer, test_dataloader, loss_fn_dict, num_param, num_param_layer, softmax, onehot_enc)
print("\n"+"="*70)
print(f'PERF FORMAT: DATASET | Acuracy: ACC | Loss: {" ".join([f"LOSS{i+1}" for i in range(num_losses)])} | PS: PS')
print("="*70)
print(f"Train | Acuracy: {train_acc['CLASSIFICATION'] *100 : 2.2f}% | Loss: { ' '.join(str(round(num, 5)) for num in train_loss) } | PS: {round(train_ps, 5)} ")
print(f"Val   | Acuracy: {val_acc['CLASSIFICATION']*100 : 2.2f}% | Loss: { ' '.join(str(round(num, 5)) for num in val_loss) } | PS: {round(val_ps, 5)} ")
print(f"Test  | Acuracy: {test_acc['CLASSIFICATION']*100 : 2.2f}% | Loss: { ' '.join(str(round(num, 5)) for num in test_loss) } | PS: {round(test_ps, 5)} ")
print("-"*70)
print(f'Optimization error  : {round(train_ps, 5)}')
print(f'Population error    : {round(test_ps, 5)}')
print(f'Generalization error: {round(test_ps  - train_ps, 5)}')

