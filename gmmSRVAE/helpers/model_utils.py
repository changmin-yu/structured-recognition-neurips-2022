from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from distributions import dirichlet, niw

torch.manual_seed(0)

def cvi_lr_schedule(lr_init, decay_rate, global_step, decay_freq):
    return lr_init * decay_rate ** (global_step/decay_freq)

def init_mm_params(num_components, latent_dims, alpha_scale=.1, beta_scale=1e-5, v_init=10., m_scale=1., C_scale=10., 
                   trainable=False, device='cuda:0'):
    alpha_init = alpha_scale * torch.ones((num_components, ))
    beta_init = beta_scale * torch.ones((num_components, ))
    v_init = torch.tile(torch.FloatTensor([latent_dims + v_init]), [num_components])
    means_init = m_scale * (2 * torch.rand((num_components, latent_dims)) -1)
    cov_init = C_scale * torch.tile(torch.eye(latent_dims).unsqueeze(0), [num_components, 1, 1])
    
    A, b, beta, v_hat = niw.standard_to_natural(beta_init, means_init, cov_init, v_init)
    alpha = dirichlet.standard_to_natural(alpha_init)
    
    params = (alpha, A, b, beta, v_hat)
    # for p in params:
    #     p = torch.tensor(p, requires_grad=trainable).to(device)
    return params

def init_mm(num_components, latent_dims, device='cuda:0', trainable_prior=False):
    theta_prior = init_mm_params(num_components, latent_dims, alpha_scale=0.05/num_components, beta_scale=.5, m_scale=0,
                                 C_scale=latent_dims+.5, v_init=latent_dims+.5, trainable=trainable_prior, device=device)
    theta = init_mm_params(num_components, latent_dims, alpha_scale=1., beta_scale=1., m_scale=5., C_scale=2*latent_dims, 
                           v_init=latent_dims+1., trainable=trainable_prior)
    return theta_prior, theta

def random_partial_isometry(M, N, std, seed=0):
    D = max(M, N)
    random_state = np.random.RandomState(seed)
    return np.linalg.qr(random_state.normal(loc=0, scale=std, size=(D, D)))[0][:M, :N].T

class View(nn.Module):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)