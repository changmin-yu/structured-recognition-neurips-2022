from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers.model_utils import random_partial_isometry

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DenseLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, std_init=.01, act=nn.ReLU()):
        super(DenseLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.act = act
        self.std_init = std_init
        self.init_params()
        
    def init_params(self):
        self.linear.weight.data.normal_(std=self.std_init)
        self.linear.bias.data.normal_(std=self.std_init)
    
    def forward(self, x):
        y = self.linear(x)
        y = self.act(y)
        return y
    
class GaussianActivation(nn.Module):
    def __init__(self, output_type='standard', dim=0):
        super().__init__()
        self.output_type = output_type
        self.dim=dim
    
    def forward(self, x, eps=1e-5):
        if self.dim:
            raw_1, raw_2 = torch.split(x, [self.dim, self.dim**2], dim=-1)
            raw_2 = torch.reshape(raw_2, (-1, self.dim, self.dim))
            raw_prec = raw_2.matmul(raw_2.transpose(-1, -2)) + eps* torch.eye(self.dim, device=DEVICE).unsqueeze(0)
            try:
                l = torch.cholesky(raw_prec)
            except:
                raw_prec = raw_prec + eps* torch.eye(self.dim, device=DEVICE).unsqueeze(0)
            raw_prec_flatten = raw_prec.reshape(-1, self.dim**2)
            if self.output_type == 'standard':
                return (raw_1, raw_prec_flatten)
            elif self.output_type == 'natparam':
                return (raw_1, -1./2 * raw_prec_flatten) # eta1 and eta2 (precision matrix)
            else:
                raise NotImplementedError
        else:
            raw_1, raw_2 = torch.chunk(x, 2, dim=-1)
        if self.output_type == 'standard':
            mean = raw_1
            var = F.softplus(raw_2)
            # return torch.cat((mean, var), dim=-1)
            return (mean, var)
        elif self.output_type == 'natparam':
            eta2 = -1./2 * F.softplus(raw_2)
            eta1 = raw_1
            # return torch.cat((eta1, eta2), dim=-1)
            return (eta1, eta2)
        else:
            raise NotImplementedError
    
class GaussianLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, output_type='standard', std_init=.01, full_dependency=False):
        super().__init__()
        self.param_type = output_type
        if full_dependency:
            self.layer = DenseLayer(in_features, out_features+out_features**2, bias=bias, act=GaussianActivation(output_type, dim=out_features), std_init=std_init)
        else:
            self.layer = DenseLayer(in_features, 2*out_features, bias=bias, act=GaussianActivation(output_type), std_init=std_init)
        
    def forward(self, x):
        return self.layer(x)
    
class ResMLP(nn.Module):
     # MLP with affine residual connection as in Johnson, et al. 2016
    def __init__(self, input_dim, hidden_dims, output_dim, std_init=.01, output_type='standard', act_fn='tanh', seed=0, 
                 full_dependency=False):
        super(ResMLP, self).__init__()
        self.input_dim = input_dim
        self.output_type = output_type
        self.output_dim = output_dim
        self.full_dependency = full_dependency
        dims = [input_dim] + hidden_dims
        if act_fn == 'tanh':
            self.act_fn = nn.Tanh()
        else:
            self.act_fn = nn.ReLU()
        self.layers = []
        for d1, d2 in zip(dims[:-1], dims[1:]):
            self.layers.extend([
                DenseLayer(d1, d2, bias=True, std_init=std_init, act=self.act_fn)
            ])
        if self.output_type == 'bernoulli':
            self.layers.extend([
                DenseLayer(dims[-1], output_dim, bias=True, std_init=std_init, act=nn.Identity())
            ])
        else:
            self.layers.extend([
                GaussianLayer(dims[-1], output_dim, bias=True, output_type=output_type, std_init=std_init, full_dependency=full_dependency)
            ])
        self.network = nn.Sequential(*self.layers)
        
        # create resnet-like residual connection
        self.W = nn.Parameter(torch.FloatTensor(random_partial_isometry(input_dim, output_dim, std_init, seed=seed)).to(DEVICE), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros((output_dim, )).to(DEVICE), requires_grad=True)
        if self.output_type != 'bernoulli':
            self.b2 = nn.Parameter(torch.zeros((output_dim, )).to(DEVICE), requires_grad=True)
            # if self.output_type == 'natparam':
            #     self.b2.data *= -0.5
    
    def forward(self, x):
        input_shape = x.shape
        assert input_shape[-1] == self.input_dim
        x = x.view(-1, self.input_dim)
        out = self.network(x)
        # return out
        out_res = F.linear(x, self.W, self.b1)
        if self.output_type != 'bernoulli':
            if self.output_type == 'standard':
                out_res_2 = torch.log1p(torch.exp(self.b2)).unsqueeze(0)
            elif self.output_type == 'natparam':
                out_res_2 = -0.5 * torch.log1p(torch.exp(self.b2)).unsqueeze(0)
            out_1, out_2 = out
            if self.full_dependency:
                out_1 = out_1 + out_res
                out_2 = out_2.reshape(-1, self.output_dim, self.output_dim) + out_res_2.unsqueeze(-1) * torch.eye(self.output_dim, device=DEVICE).unsqueeze(0)
                l = torch.cholesky(-2. * out_2)
            else:
                out_1 = out_1 + out_res
                out_2 = out_2 + out_res_2
            out_1 = out_1.view(input_shape[:-1] + (self.output_dim, ))
            if self.full_dependency:
                out_2 = out_2.view(input_shape[:-1] + (self.output_dim, self.output_dim))
            else:
                out_2 = out_2.view(input_shape[:-1] + (self.output_dim, ))
            out_new = (out_1, out_2)
            # out[..., :self.output_dim//2] += out_res
            # out[..., self.output_dim//2:] += out_res_2
        else:
            out_new = out + out_res
            out_new = out_new.view(input_shape[:-1] + (self.output_dim, ))
        return out_new