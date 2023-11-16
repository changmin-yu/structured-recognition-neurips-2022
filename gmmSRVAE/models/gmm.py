from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from math import inf

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable # convert this into a class 
from torch.distributions.dirichlet import Dirichlet
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import PIL.Image
from torchvision.transforms import ToTensor

from helpers.data import make_dataloader
from distributions import dirichlet, niw
from helpers.general_utils import generate_log_id, generate_param_schedule
from helpers.model_utils import init_mm_params
from helpers.losses import weighted_mse
from helpers.visualise_gmm import plot_clusters


'''
Variational approximation with mixture of Gaussians, in Chapter 10.2 of Pattern Recognition and Machine Learning, Christopher M. Bishop, 2006
'''

def update_Nk(r_nk):
    return torch.sum(r_nk, dim=0) # (N, K) -> (K, )

def update_xk(x, r_nk, N_k):
    x_k = torch.einsum('nk,nd->kd', r_nk, x)
    x_k_normalised = x_k / (N_k.unsqueeze(-1))
    return torch.where(torch.isnan(x_k_normalised), x_k, x_k_normalised)

def update_Sk(x, r_nk, N_k, x_k):
    x_xk = x.unsqueeze(1) - x_k.unsqueeze(0)
    S = torch.einsum('nk,nkde->kde', r_nk, torch.einsum('nkd,nke->nkde', x_xk, x_xk))
    S_normed = S / N_k.unsqueeze(-1).unsqueeze(-1)
    return torch.where(torch.isnan(S_normed), S, S_normed)

def update_alphak(alpha_0, N_k):
    return alpha_0 + N_k

def update_betak(beta_0, N_k):
    return beta_0 + N_k

def update_mk(beta_0, beta_k, m_0, x_k, N_k):
    if len(beta_0.shape) == 1:
        beta_0 = beta_0.unsqueeze(-1)
    return 1/beta_k.unsqueeze(-1) * (beta_0 * m_0 + N_k.unsqueeze(-1) * x_k)

def update_Wk(W_0, x_k, N_k, S_k, m_0, beta_0, beta_k):
    W_k = W_0 + N_k.unsqueeze(-1).unsqueeze(-1)* S_k
    diff = x_k - m_0
    W_k = W_k + torch.einsum('k,kij->kij', beta_0 * N_k / beta_k, torch.einsum('ki,kj->kij', diff, diff))
    return W_k

def update_vk(v_0, N_k):
    return v_0 + N_k + 1

def expected_mahalanobis_dist(X, m_k, W_k, v_k, beta_k):
    N, D = X.shape
    diff = X.unsqueeze(1) - m_k.unsqueeze(0)
    squared_error = v_k.unsqueeze(0) * torch.einsum('ndk,nkd->nk', torch.transpose(diff, 1, 2), torch.einsum('kde,nke->nkd', W_k, diff))
    return D/beta_k.unsqueeze(0) + squared_error

def expected_mahalanobis_dist_imputation(X, m_k, W_k, v_k, beta_k, data_mask):
    N, D = X.shape
    diff = X.unsqueeze(1) - m_k.unsqueeze(0)
    data_mask = (1.-torch.FloatTensor(data_mask)).unsqueeze(1)
    diff_mask = diff * data_mask

    squared_error = v_k.unsqueeze(0) * torch.einsum('nkd,nkd->nk', diff_mask, 
                                                    torch.einsum('kde,nke->nkd', W_k, diff_mask))
    return D/beta_k.unsqueeze(0) + squared_error

def expected_logdet_precision(v_k, W_k):
    K, D, _ = W_k.shape
    det_W = torch.linalg.det(W_k)
    logdet_W = torch.where(det_W > 1e-20, torch.log(det_W), torch.zeros_like(det_W, dtype=torch.float32))
    exp_logdet = torch.sum(torch.digamma((v_k.unsqueeze(-1) + 1 - torch.arange(D).unsqueeze(0))/2), 
                           dim=1) + D * torch.log(torch.tensor(2)) + logdet_W
    return exp_logdet

def expected_log_pi(alpha_k):
    return torch.digamma(alpha_k) - torch.digamma(torch.sum(alpha_k))

def update_rnk(exp_log_pi, exp_logdet_cov, exp_dist):
    log_rho_nk = exp_log_pi.unsqueeze(0) + 1/2*exp_logdet_cov.unsqueeze(0) - 0.5*exp_dist
    rho_nk = torch.exp(log_rho_nk - torch.max(log_rho_nk, dim=1, keepdim=True)[0])
    normaliser = torch.sum(rho_nk, dim=1, keepdim=True)
    return rho_nk / normaliser

def e_step(X, alpha_k, beta_k, v_k, W_k, m_k):
    exp_dist = expected_mahalanobis_dist(X, m_k, W_k, v_k, beta_k)
    exp_log_pi = expected_log_pi(alpha_k)
    exp_logdet_prec = expected_logdet_precision(v_k, W_k)
    r_nk = update_rnk(exp_log_pi, exp_logdet_prec, exp_dist)
    return r_nk, exp_log_pi

def e_step_imputation(X, alpha_k, beta_k, v_k, W_k, m_k, data_mask):
    exp_dist = expected_mahalanobis_dist_imputation(X, m_k, W_k, v_k, beta_k, data_mask)
    exp_log_pi = expected_log_pi(alpha_k)
    exp_logdet_prec = expected_logdet_precision(v_k, W_k)
    r_nk = update_rnk(exp_log_pi, exp_logdet_prec, exp_dist)
    return r_nk, exp_log_pi

def m_step(X, r_nk, alpha_0, beta_0, m_0, W_0, v_0):
    N_k = update_Nk(r_nk)
    x_k = update_xk(X, r_nk, N_k)
    S_k = update_Sk(X, r_nk, N_k, x_k)
    alpha_k = update_alphak(alpha_0, N_k)
    beta_k = update_betak(beta_0, N_k)
    m_k = update_mk(beta_0, beta_k, m_0, x_k, N_k)
    W_k = update_Wk(W_0, x_k, N_k, S_k, m_0, beta_0, beta_k)
    v_k = update_vk(v_0, N_k)
    return alpha_k, beta_k, m_k, W_k, v_k, x_k, S_k

def inference(X, K, seed=0):
    torch.manual_seed(seed)
    N, D = X.shape
    r_nk = torch.tensor(Dirichlet(torch.ones(K)).sample(N), dtype=torch.float32, requires_grad=True)
    alpha, A, b, beta, v_hat = init_mm_params(K, D, alpha_scale=.5/K, beta_scale=0.5, m_scale=0, C_scale=D+0.5, 
                                              v_init=D+0.5, trainable=False)
    beta_0, m_0, W_0, v_0 = niw.natural_to_standard(A, b, beta, v_hat)
    alpha_0 = dirichlet.natural_to_standard(alpha)
    
    alpha_k, beta_k, m_k, W_k, v_k, x_k, S_k = m_step(X, r_nk, alpha_0, beta_0, m_0, W_0, v_0)
    W_k_inv = torch.linalg.inv(W_k)
    r_nk_new, exp_log_pi = e_step(X, alpha_k, beta_k, v_k, W_k_inv, m_k)
    theta = (alpha_k, beta_k, m_k, W_k, v_k) # updated global parameters
    log_r_nk = torch.log(r_nk_new)
    return r_nk_new, log_r_nk, theta, (x_k, S_k, exp_log_pi)

def variational_free_energy(N_k, r_nk, log_pi, alpha_0, logdet_prec, beta_k, S_k, W_k, v_k, x_k, m_k, beta_0, v_0):
    pass

def variational_EM_gmm(X_train, X_test, writer, K=10, num_iter=500, threshold=1e-3, seed=1, eval_freq=10):
    N, D = X_train.shape
    N_test, _ = X_test.shape

    pred_error = inf
    i = 0

    # initialise
    r_nk = torch.tensor(Dirichlet(torch.ones(K)).sample((N, )), dtype=torch.float32, requires_grad=False)
    A, b, beta, v_hat, alpha = init_mm_params(K, D, alpha_scale=.5/K, beta_scale=.5, m_scale=0., C_scale=D+0.5, 
                                                v_init=D+0.5, trainable=False)
    beta_0, m_0, W_0, v_0 = niw.natural_to_standard(A, b, beta, v_hat)
    alpha_0 = dirichlet.natural_to_standard(alpha)
    # alpha_k = alpha_0.clone()
    # beta_k = beta_0.clone()
    # v_k = v_0.clone()
    # W_k = W_0.clone()
    # W_k_inv = torch.linalg.inv(W_k).clone()
    # m_k = m_0.clone()
    while i < num_iter and pred_error > threshold:
        alpha_k, beta_k, m_k, W_k, v_k, x_k, S_k = m_step(X_train, r_nk, alpha_0, beta_0, m_0, W_0, v_0)
        W_k_inv = torch.linalg.inv(W_k)
        r_nk_new, exp_log_pi = e_step(X_train, alpha_k, beta_k, v_k, W_k_inv, m_k)
        log_r_nk = torch.log(r_nk_new).clone()
        r_nk = r_nk_new.clone()
        
        theta = (alpha_k, beta_k, v_k, W_k_inv, m_k)
        
        pred_x_means = torch.tile(x_k.unsqueeze(0).unsqueeze(2), (N, 1, 1, 1))
        mse_train = weighted_mse(X_train, pred_x_means, r_nk_new)
        writer.add_scalar('mse-train', mse_train, i+1)
        
        print(f'{i} iterations done!')
        
        if (i+1) % eval_freq == 0:
            r_nk_test, _ = e_step(X_test, *theta)
            pred_x_means_test = torch.tile(x_k.unsqueeze(0).unsqueeze(2), (N_test, 1, 1, 1))
            mse_test = weighted_mse(X_test, pred_x_means_test, r_nk_test)
            writer.add_scalar('mse-test', mse_test, i+1)
            
            f, ax = plt.subplots()
            image = plot_clusters(X_train, x_k, S_k, log_r_nk, exp_log_pi, ax)
            writer.add_image('cluster-results', image, i+1)
            
            pred_error = mse_test
            print(f'{i} iterations done! | test-mse: {pred_error:.2f}')
        
        i += 1
    return i, r_nk, log_r_nk, theta, (x_k, S_k, exp_log_pi)

if __name__=='__main__':
    datapath = './datasets'
    ratio_train = 0.9
    ratio_val = None
    ratio_test = 0.1
    num_iter = 1000
    eval_freq = 10
    K = 10
    seed = 0
    seed_data = 0
    logdir = './logdir'

    schedule = generate_param_schedule({
        'method': 'gmm', 
        'dataset': 'pinwheel', 
        'K': 8, 
        'seed': 0
    })
    
    for config_id, config in enumerate(schedule):
        K = config['K']
        seed = config['seed']
        dataset = config['dataset']
        print(f'Experiment {config_id} with config: \n{config}')
        
        log_id = generate_log_id(config)
        logdir = os.path.join(logdir, log_id)
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        writer = SummaryWriter(logdir)
        
        torch.manual_seed(seed)
        train_loader, test_loader, _ = make_dataloader(dataset, ratio_train=ratio_train, ratio_val=None, 
                                                                datapath=datapath, batch_size=-1, test_bs=-1, n_towers=1, 
                                                                seed_split=seed_data, seed_minibatch=seed_data)
        train_loader = iter(train_loader)
        test_loader = iter(test_loader)
        X_train, y_train = next(train_loader)
        X_test, y_test = next(test_loader)
        
        i, r_nk, log_r_nk, theta, stats = variational_EM_gmm(X_train, X_test, writer, K=K, num_iter=num_iter, threshold=0.001, 
                                                             seed=seed, eval_freq=eval_freq)
