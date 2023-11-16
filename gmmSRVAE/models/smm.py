from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from math import inf
import numpy as np
import torch
from torch.distributions import Dirichlet
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from helpers.data import make_dataloader
from distributions import dirichlet, niw
from helpers.general_utils import generate_log_id, generate_param_schedule, logdet
from helpers.losses import weighted_mse
from helpers.visualise_gmm import plot_clusters
from helpers.model_utils import init_mm_params

def update_Nk(r_nk):
    return torch.sum(r_nk, dim=0)

def update_Wk(ru_nk):
    return torch.sum(ru_nk, dim=0)

def update_xk(x, W_k, ru_nk):
    return 1/(W_k.unsqueeze(-1) + 1e-10) * torch.einsum('nk,nd->kd', ru_nk, x)

def update_Sk(x, ru_nk, x_k, W_k):
    err = x.unsqueeze(1) - x_k.unsqueeze(0)
    dist = torch.einsum('nkd,nke->nkde', err, err)
    S_k = torch.einsum('nk,nkde->kde', ru_nk, dist)
    S_k = S_k / (W_k.unsqueeze(-1).unsqueeze(-1) + 1e-10)
    return S_k

def update_alphak(alpha0, N_k):
    return alpha0 + N_k

def update_betak(beta0, W_k):
    return beta0 + W_k

def update_mk(beta0, m0, betak, x_k, W_k):
    if len(beta0.shape) == 1:
        beta0 = beta0.unsqueeze(-1)
    mk = (W_k.unsqueeze(-1) * x_k + beta0 * m0) / betak.unsqueeze(-1)
    return mk

def update_vk(N_k, v0):
    return N_k + v0

def update_Ck(W_0, x_k, m0, W_k, beta0, betak, S_k):
    Ck = W_k.unsqueeze(1).unsqueeze(-1) * S_k + W_0
    err = x_k - m0
    Ck = Ck + torch.einsum('k,kde->kde', W_k * beta0 / betak, torch.einsum('kd,ke->kde', err, err))
    return Ck

def exp_square_dist(X, m_k, P_k, v_k, beta_k, kappa_k):
    N, D = X.shape
    err = X.unsqueeze(1) - m_k.unsqueeze(0)
    dist = torch.einsum('nkd,nkd->nk', err, torch.einsum('kde,nke->nkd', P_k, err))
    dist = (beta_k/kappa_k).unsqueeze(0) * dist + D / (kappa_k * v_k).unsqueeze(0) + 1
    return dist

def exp_logdet_prec(v_k, P_k):
    K, D, _ = P_k.shape
    logdet_P = logdet(P_k)
    exp_logdet = torch.sum(torch.digamma((v_k.unsqueeze(-1) + 1 - torch.arange(D).unsqueeze(0))/2), 
                           dim=1) + D * torch.log(torch.tensor(2)) + logdet_P
    return exp_logdet

def exp_log_pi(alpha_k):
    alpha = torch.digamma(alpha_k) - torch.digamma(alpha_k.sum())
    return alpha

def compute_r_nk(X, m_k, P_k, v_k, beta_k, alpha_k, kappa_k):
    N, D = X.shape
    expt_log_pi = exp_log_pi(alpha_k)
    expt_logdet_prec = exp_logdet_prec(v_k, P_k)
    expt_square_dist = exp_square_dist(X, m_k, P_k, v_k, beta_k, kappa_k)
    log_r_nk = expt_log_pi.unsqueeze(0) + .5 * expt_logdet_prec.unsqueeze(0) - (D+kappa_k.unsqueeze(0))/2 * expt_square_dist
    log_r_nk = log_r_nk + torch.lgamma((D+kappa_k.unsqueeze(0))/2) - torch.lgamma(kappa_k.unsqueeze(0)/2) - 0.5*D*torch.log(kappa_k.unsqueeze(0) * np.pi)
    normaliser = torch.logsumexp(log_r_nk, dim=-1, keepdim=True)
    return torch.exp(log_r_nk - normaliser)

def expect_u_nk(X, kappa_k, beta_k, m_k, P_k, v_k):
    N, D = X.shape
    alpha_nk = 0.5 * (D + kappa_k)
    err = X.unsqueeze(1) - m_k.unsqueeze(0)
    dist = torch.einsum('nkd,nkd->nk', err, torch.einsum('kde,nke->nkd', P_k, err))
    beta_nk = 0.5 * (beta_k.unsqueeze(0) * dist + (D * kappa_k / v_k).unsqueeze(0) + kappa_k.unsqueeze(0))
    return alpha_nk / beta_nk

def e_step(X, alpha_k, beta_k, m_k, P_k, v_k, kappa_k):
    r_nk = compute_r_nk(X, m_k, P_k, v_k, beta_k, alpha_k, kappa_k)
    u_nk = expect_u_nk(X, kappa_k, beta_k, m_k, P_k, v_k)
    log_pi = exp_log_pi(alpha_k)
    return r_nk, u_nk, log_pi.exp()

def m_step(X, r_nk, u_nk, alpha_0, beta_0, m_0, C_0, v_0):
    ru_nk = r_nk * u_nk
    N_k = update_Nk(r_nk)
    W_k = update_Wk(ru_nk)
    x_k = update_xk(X, W_k, ru_nk)
    S_k = update_Sk(X, ru_nk, x_k, W_k)
    
    alpha_k = update_alphak(alpha_0, N_k)
    beta_k = update_betak(beta_0, W_k)
    m_k = update_mk(beta_0, m_0, beta_k, x_k, W_k)
    C_k = update_Ck(C_0, x_k, m_0, W_k, beta_0, beta_k, S_k)
    v_k = update_vk(N_k, v_0)
    return alpha_k, beta_k, m_k, C_k, v_k, x_k, S_k

def variational_EM_smm(X_train, X_test, writer, kappa_init, K=10, num_iter=500, threshold=1e-3, seed=1, eval_freq=10):
    N, D = X_train.shape
    N_test, _ = X_test.shape

    pred_error = inf
    i = 0

    # initialise
    r_nk = torch.tensor(Dirichlet(torch.ones(K)).sample((N, )), dtype=torch.float32, requires_grad=False)
    u_nk = torch.ones_like(r_nk, dtype=torch.float32)
    kappa_k = torch.ones((K, )) * kappa_init
    A, b, beta, v_hat, alpha = init_mm_params(K, D, alpha_scale=.05/K, beta_scale=.5, m_scale=0., C_scale=D+0.5, 
                                                v_init=D+0.5, trainable=False)
    beta_0, m_0, C_0, v_0 = niw.natural_to_standard(A, b, beta, v_hat)
    alpha_0 = dirichlet.natural_to_standard(alpha)

    while i < num_iter and pred_error > threshold:
        alpha_k, beta_k, m_k, C_k, v_k, x_k, S_k = m_step(X_train, r_nk, u_nk, alpha_0, beta_0, m_0, C_0, v_0)
        C_k_inv = torch.linalg.inv(C_k)
        r_nk_new, u_nk_new, exp_pi = e_step(X_train, alpha_k, beta_k, m_k, C_k_inv, v_k, kappa_k)
        log_r_nk = torch.log(r_nk_new).clone()
        r_nk = r_nk_new.clone()
        u_nk = u_nk_new.clone()
        
        theta = (alpha_k, beta_k, m_k, C_k_inv, v_k, kappa_k)
        
        pred_x_means = torch.tile(x_k.unsqueeze(0).unsqueeze(2), (N, 1, 1, 1))
        mse_train = weighted_mse(X_train, pred_x_means, r_nk_new)
        writer.add_scalar('mse-train', mse_train, i+1)
        
        print(f'{i} iterations done!')
        
        if (i+1) % eval_freq == 0:
            r_nk_test, _, _ = e_step(X_test, *theta)
            pred_x_means_test = torch.tile(x_k.unsqueeze(0).unsqueeze(2), (N_test, 1, 1, 1))
            mse_test = weighted_mse(X_test, pred_x_means_test, r_nk_test)
            writer.add_scalar('mse-test', mse_test, i+1)
            
            f, ax = plt.subplots()
            image = plot_clusters(X_train, x_k, S_k, log_r_nk, exp_pi, ax)
            writer.add_image('cluster-results', image, i+1)
            
            pred_error = mse_test
            print(f'{i} iterations done! | test-mse: {pred_error:.2f}')
        
        i += 1
    return i, r_nk, u_nk, log_r_nk, theta, (x_k, S_k, exp_pi)

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
        'kappa': 1000,
        'seed': 0
    })
    
    for config_id, config in enumerate(schedule):
        K = config['K']
        seed = config['seed']
        dataset = config['dataset']
        kappa_init = config['kappa']
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
        
        i, r_nk, log_r_nk, theta, stats = variational_EM_smm(X_train, X_test, writer, kappa_init, K=K, num_iter=num_iter, threshold=0.001, 
                                                             seed=seed, eval_freq=eval_freq)
