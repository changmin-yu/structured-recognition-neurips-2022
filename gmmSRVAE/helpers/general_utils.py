from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from itertools import product
import torch
import numpy as np
import argparse

def parse_args():
    args = argparse.ArgumentParser(description='args for SVAE-CVI test')
    args.add_argument('--dataset', default='auto', type=str, 
                      help='dataset to use')
    args.add_argument('--datapath', default='./datasets', type=str, 
                      help='path to datasets')
    args.add_argument('--train-ratio', default=0.9, type=float, 
                      help='training ratio in loading the dataset')
    args.add_argument('--num-iters', default=100000, type=int, 
                      help='number of iterations')
    args.add_argument('--train-bs', default=128, type=int, 
                      help='training batch size')
    args.add_argument('--test-bs', default=128, type=int, 
                      help='testing batch size')
    args.add_argument('--num-latent', default=10, type=int, 
                      help='number of latent samples to use for elbo computation (IWAE)')
    args.add_argument('--eval-freq', default=1, type=int, 
                      help='evaluation frequency (epochs)')
    args.add_argument('--num-mixtures', default=10, type=int, 
                      help='number of mixtures')
    args.add_argument('--seed', default=0, type=int, 
                      help='random seed')
    args.add_argument('--logdir', default='./logdir', type=str, 
                      help='logging directory')
    args.add_argument('--model', default='svae-cvi', type=str, 
                      help='training model type')
    args.add_argument('--lr', default=3e-4, type=float, 
                      help='learning rate of the recognition and generative parameters')
    args.add_argument('--lr-cvi', default=0.2, type=float, 
                      help='learning rate of CVI updates for the PGM parameters')
    args.add_argument('--latent-dim', default=6, type=int, 
                      help='dimension of latent codes')
    args.add_argument('--hidden-dim', default=50, type=int, 
                      help='number of hidden units in each layers')
    args.add_argument('--num-hidden', default=2, type=int, 
                      help='number of hidden layers (note that we assume identical network structure for the\
                          recognition and generative networks')
    args.add_argument('--act-fn', default='tanh', type=str, 
                      help='activation function to use in the vaes')
    args.add_argument('--full-dependency', default=False, type=bool, 
                      help='whether or not the recognition network outputs full covariance matrix')
    return args.parse_args()

def generate_log_id_new(parser):
    return f'{parser.model}_{parser.dataset}_lr_{parser.lr}_L_{parser.num_latent}_S_{parser.seed}_A_{parser.act_fn}_M_{parser.num_mixtures}_FD_{parser.full_dependency}'

def generate_log_id(config, method_key='method', dataset_key='dataset'):
    method = config.get(method_key, 'unknownM')
    dataset = config.get(dataset_key, 'unknownD')
    log_id = f'{method}_{dataset}'
    for key, val in iter(sorted(config.items())):
        if key != method_key and key != dataset_key:
            if isinstance(val, str):
                val_str = val
            elif isinstance(val, int):
                val_str = f'{val}'
            elif isinstance(val, float):
                val_str = f'{val:.5f}'
            else:
                raise NotImplementedError
            log_id += f'_{key}_{val_str}'
    
    return log_id

def generate_param_schedule(param_ranges, verbose=False):
    param_lists = []
    for param, vals in param_ranges.items():
        if isinstance(vals, str):
            vals = [vals]
        elif not isinstance(vals, collections.Iterable):
            vals = [vals]
        param_lists.append([(param, v) for v in vals])
    
    schedule = [dict(p) for p in product(*param_lists)]
    print(f'Created parameter scheduling containing {len(schedule):d} configurations')
    if verbose:
        for config in schedule:
            print(config)
        print('-----------------------------------------')
    return schedule
    
def logdet(A):
    return 2. * torch.log(torch.diagonal(torch.linalg.cholesky(A), dim1=-2, dim2=-1)).sum(-1)

def outer(x, y):
    x_ = x.unsqueeze(-1)
    y_ = y.unsqueeze(-2)
    return x_*y_

def gather_nd(params, indices):
    orig_shape = list(indices.shape)
    num_samples = np.prod(orig_shape[:-1])
    m = orig_shape[-1]
    n = len(params.shape)

    if m <= n:
        out_shape = orig_shape[:-1] + list(params.shape)[m:]
    else:
        raise ValueError(
            f'the last dimension of indices must less or equal to the rank of params. Got indices:{indices.shape}, params:{params.shape}. {m} > {n}'
        )

    indices = indices.reshape((num_samples, m)).transpose(0, 1).tolist()
    output = params[indices]    # (num_samples, ...)
    return output.reshape(out_shape).contiguous()