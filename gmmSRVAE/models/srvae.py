from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, MultivariateNormal
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import copy

from distributions import dirichlet, gaussian, niw, student_t
from models import gmm, smm
from models.networks import ResMLP
from helpers.general_utils import gather_nd, generate_log_id_new, parse_args
from helpers.model_utils import init_mm, cvi_lr_schedule
from helpers.data import make_dataloader
from helpers.losses import weighted_mse

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# torch.autograd.set_detect_anomaly(True) # comment this line out after debugging

class SVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, enc_hidden_dims, dec_hidden_dims, std_init, 
                 num_mixture=6, num_latent_sample=10, enc_output_type='natparam', 
                 dec_output_type='standard', pgm_type='gmm', act_fn='tanh', dof=5, 
                 full_dependency=False):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.enc_output_type = enc_output_type
        self.dec_output_type = dec_output_type
        self.pgm_type = pgm_type
        self.num_latent_sample = num_latent_sample
        self.num_mixture = num_mixture
        self.dof = 5
        self.full_dependency = full_dependency
        self.recognition_network = ResMLP(input_dim, enc_hidden_dims, latent_dim, std_init, 
                                        output_type=enc_output_type, act_fn=act_fn, full_dependency=full_dependency)
        self.generative_network = ResMLP(latent_dim, dec_hidden_dims, input_dim, std_init, 
                                         output_type=dec_output_type, act_fn=act_fn)
        self.epsilon_distribution = Normal(torch.tensor(0.0).to(DEVICE), torch.tensor(1.0).to(DEVICE))
        self.init_phi()
        
    def init_phi(self):
        with torch.no_grad():
            if self.pgm_type == 'smm':
                gmm_prior, theta = init_mm(self.num_mixture, self.latent_dim, DEVICE)
                theta_standard = niw.natural_to_standard(*gmm_prior)
                mu_k_init, sigma_k = niw.expectation(*theta_standard)
                L_k_init = torch.linalg.cholesky(sigma_k)
                dof = torch.ones((self.num_mixture, ), dtype=torch.float32) * self.dof
                alpha_k = theta[0]
                
                phi_gmm = self.init_recognition_params(theta, self.num_mixture)
                
                gmm_prior = gmm_prior[0]
                theta = (alpha_k, mu_k_init, L_k_init, dof)
            else:
                gmm_prior, theta = init_mm(self.num_mixture, self.latent_dim, DEVICE)
                phi_gmm = self.init_recognition_params(theta)
            self.theta = tuple([t.to(DEVICE) for t in theta])
            self.pgm_prior = tuple([t.to(DEVICE) for t in gmm_prior])
            phi_gmm = tuple([t.to(DEVICE) for t in phi_gmm])
            self.mu_k_recog = nn.Parameter(phi_gmm[0].to(DEVICE), requires_grad=True)
            self.L_k_recog = nn.Parameter(phi_gmm[1].to(DEVICE), requires_grad=True)
            self.pi_k_recog = nn.Parameter(phi_gmm[2].to(DEVICE), requires_grad=True)
            self.phi_gmm = (self.mu_k_recog, self.L_k_recog, self.pi_k_recog)
    
    def init_recognition_params(self, theta):
        pi_k_init = F.softmax(torch.randn((self.num_mixture, )), dim=0)
        mu_k, L_k = self.make_loc_scale_params(theta)
        return (mu_k, L_k, pi_k_init)
    
    def make_loc_scale_params(self, theta):
        theta_standard = niw.natural_to_standard(*theta[1:])
        mu_k_init, sigma_k_init = niw.expectation(*theta_standard)
        L_k_init = torch.linalg.cholesky(sigma_k_init)
        return mu_k_init, L_k_init
        
    def recognition_potential(self, x):
        return self.recognition_network(x)
    
    def generation(self, z):
        out = self.generative_network(z)
        if self.dec_output_type == 'bernoulli':
            out = F.sigmoid(out)
        return out
    
    def forward(self, y):
        phi_recognition = self.recognition_potential(y)
        X_samples, log_z_given_y_phi, phi_tilde, dbg = self.e_step(phi_recognition)
        y_recon = self.generation(X_samples)
        X_subsamples = subsample_x(X_samples, log_z_given_y_phi)[:, 0, :]
        return y_recon, phi_recognition, X_samples, X_subsamples, log_z_given_y_phi, phi_tilde

    def elbo(self, y):
        y_recon, phi_recognition, X_samples, X_subsamples, log_z_given_y_phi, phi_tilde = self(y)
        
        if self.pgm_type == 'gmm':
            with torch.no_grad():
                beta_k, m_k, C_k, v_k = niw.natural_to_standard(*self.theta[1:])
                mu, sigma = niw.expectation(beta_k, m_k, C_k, v_k)
                eta1_theta, eta2_theta = gaussian.standard_to_natural(mu, sigma)
                alpha_k = dirichlet.natural_to_standard(self.theta[0])
                expected_log_pi_theta = dirichlet.expected_log_pi(alpha_k)
        elif self.pgm_type == 'smm':
            with torch.no_grad():
                mu_theta, sigma_theta = unpack_smm(self.theta[1:3])
                alpha_k = dirichlet.natural_to_standard(self.theta[0])
                expected_log_pi_theta = dirichlet.expected_log_pi(alpha_k)
                dof = self.theta[3]
        else:
            raise NotImplementedError
        
        r_nk = log_z_given_y_phi.exp()
        if self.dec_output_type == 'bernoulli':
            recon_loss = weighted_bernoulli_loglike(y, y_recon, r_nk)
        elif self.dec_output_type == 'standard':
            mean_pred, var_pred = y_recon
            recon_loss = weighted_diag_gaussian_loglike(y, mean_pred, var_pred, r_nk)
        else:
            raise NotImplementedError
        
        N = X_samples.shape[0]
        eta1_phi_tilde, eta2_phi_tilde = phi_tilde
        eta1_phi_tilde = eta1_phi_tilde.squeeze(-1)
        
        logprob_x_given_phi = gaussian.log_prob_nat_per_sample(X_samples, eta1_phi_tilde, eta2_phi_tilde)
        log_numerator = logprob_x_given_phi + log_z_given_y_phi.unsqueeze(-1) # (N, K, S)
        
        if self.pgm_type == 'gmm':
            logprob_x_given_theta = gaussian.log_prob_nat_per_sample(X_samples, torch.tile(eta1_theta.unsqueeze(0), [N, 1, 1]), 
                                                                    torch.tile(eta2_theta.unsqueeze(0), [N, 1, 1, 1]))
        elif self.pgm_type == 'smm':
            logprob_x_given_theta = student_t.logprob_per_sample(X_samples, mu_theta, sigma_theta, dof)
        log_denominator = logprob_x_given_theta + expected_log_pi_theta.unsqueeze(0).unsqueeze(-1)
        
        kl_div = torch.mean(torch.sum(r_nk.unsqueeze(-1) * (log_numerator - log_denominator), dim=1))
        
        elbo = -(recon_loss - kl_div)
        dbg = (recon_loss, torch.sum(r_nk * log_numerator.mean(-1)), torch.sum(r_nk * log_denominator.mean(-1)), kl_div)
        
        return elbo, dbg, (X_subsamples, log_z_given_y_phi)

    def e_step(self, phi_enc):
        eta1_phi1, eta2_phi1 = phi_enc
        if not self.full_dependency:
            eta2_phi1 = torch.eye(eta2_phi1.shape[-1]).unsqueeze(0).to(DEVICE) * torch.tile(eta2_phi1.unsqueeze(-1), [1, 1, eta2_phi1.shape[-1]])
        
        eta1_phi2, eta2_phi2, pi_phi2 = unpack_recognition_gmm(self.phi_gmm)
        log_z_given_y_phi, dbg = compute_log_z_given_y(eta1_phi1, eta2_phi1, eta1_phi2, eta2_phi2, pi_phi2)
        
        eta1_phi_tilde = (eta1_phi1.unsqueeze(1) + eta1_phi2.unsqueeze(0)).unsqueeze(-1)
        eta2_phi_tilde = eta2_phi1.unsqueeze(1) + eta2_phi2.unsqueeze(0)
        phi_tilde = (eta1_phi_tilde, eta2_phi_tilde)
        
        x_samples = sample_x_per_comp(eta1_phi_tilde, eta2_phi_tilde, self.num_latent_sample)
        return x_samples, log_z_given_y_phi, phi_tilde, dbg

    def m_step(self, X_samples, res_nk):
        if self.pgm_type == 'gmm':
            # the same m-step as in standard variational gmm
            beta_0, m_0, C_0, v_0 = niw.natural_to_standard(*self.pgm_prior[1:])
            alpha_0 = dirichlet.natural_to_standard(self.pgm_prior[0])
            alpha_k, beta_k, m_k, C_k, v_k, x_k, S_k = gmm.m_step(X_samples, res_nk, alpha_0, beta_0, m_0, C_0, v_0)
            A, b, beta, v_hat =niw.standard_to_natural(beta_k, m_k, C_k, v_k)
            alpha = dirichlet.standard_to_natural(alpha_k)
            return (alpha, A, b, beta, v_hat)
        elif self.pgm_type == 'smm':
            alpha_0 = dirichlet.natural_to_standard(self.pgm_prior[0])
            N_k = smm.update_Nk(res_nk)
            alpha_k = smm.update_alphak(alpha_0, N_k)
            alpha = dirichlet.standard_to_natural(alpha_k)
            return alpha
        
    def update_gmm_params(self, theta_new, step_size):
        theta_old = copy.deepcopy(self.theta)
        if self.pgm_type == 'smm':
            theta_old = [self.theta[0]]
        updates = []
        for i, (curr_param, new_param) in enumerate(zip(theta_old, theta_new)):
            updates.append((1-step_size) * curr_param + step_size * new_param)
        if self.pgm_type == 'smm':
            self.theta[0] = updates[0]
        elif self.pgm_type == 'gmm':
            self.theta = copy.deepcopy(updates)
        else:
            raise NotImplementedError
        return updates

def m_step_smm(smm_prior, res_nk):
    # we are only updating the alpha parameters in this case
    alpha_0 = dirichlet.natural_to_standard(smm_prior[0])
    N_k = smm.update_Nk(res_nk)
    alpha_k = smm.update_alphak(alpha_0, N_k)
    alpha = dirichlet.standard_to_natural(alpha_k)
    return alpha

def compute_elbo_gmm(X, X_recon, theta, phi_tilde, X_samples, log_z_given_y_phi, decoder_type='standard'):
    beta_k, m_k, C_k, v_k = niw.natural_to_standard(*theta[1:])
    mu, sigma = niw.expectation(beta_k, m_k, C_k, v_k)
    eta1_theta, eta2_theta = gaussian.standard_to_natural(mu, sigma)
    alpha_k = dirichlet.natural_to_standard(theta[0])
    expected_log_pi_theta = dirichlet.expected_log_pi(alpha_k)
    
    # stop the gradient from updating the PGM natural parameters
    eta1_theta = nn.Parameter(eta1_theta, requires_grad=False)
    eta2_theta = nn.Parameter(eta2_theta, requires_grad=False)
    expected_log_pi_theta = nn.Parameter(expected_log_pi_theta, requires_grad=False)
    
    r_nk = log_z_given_y_phi.exp()
    if decoder_type == 'bernoulli':
        recon_loss = weighted_bernoulli_loglike(X, X_recon, r_nk)
    elif decoder_type == 'standard':
        mean_pred, var_pred = X_recon
        recon_loss = weighted_diag_gaussian_loglike(X, mean_pred, var_pred, r_nk)
    else:
        raise NotImplementedError
    
    # E[log q(x, z=k|y, phi)] note that this is not technically the kl-div, rather the entropy of q
    N = X_samples.shape[0]
    eta1_phi_tilde, eta2_phi_tilde = phi_tilde
    eta1_phi_tilde = eta1_phi_tilde.squeeze(-1)
    
    logprob_x_given_phi = gaussian.log_prob_nat_per_sample(X_samples, eta1_phi_tilde, eta2_phi_tilde)
    log_numerator = logprob_x_given_phi + log_z_given_y_phi.unsqueeze(-1) # (N, K, S)
    
    logprob_x_given_theta = gaussian.log_prob_nat_per_sample(X_samples, torch.tile(eta1_theta.unsqueeze(0), [N, 1, 1]), 
                                                             torch.tile(eta2_theta.unsqueeze(0), [N, 1, 1, 1]))
    log_denominator = logprob_x_given_theta + expected_log_pi_theta.unsqueeze(0).unsqueeze(-1)
    
    kl_div = torch.mean(torch.sum(r_nk.unsqueeze(-1) * (log_numerator - log_denominator), dim=1))
    
    elbo = recon_loss - kl_div
    dbg = (recon_loss, torch.sum(r_nk * log_numerator.mean(-1)), torch.sum(r_nk * log_denominator.mean(-1)), kl_div)
    
    return elbo, dbg

def compute_elbo_smm(X, X_recon, theta, phi_tilde, X_samples, log_z_given_y_phi, decoder_type='standard'):
    mu_theta, sigma_theta = unpack_smm(theta[1:3])
    alpha_k = dirichlet.natural_to_standard(theta[0])
    expected_log_pi_theta = dirichlet.expected_log_pi(alpha_k)
    dof = theta[3]
    
    expected_log_pi_theta = nn.Parameter(expected_log_pi_theta, requires_grad=False)
    dof = nn.Parameter(dof, requires_grad=False)
    
    r_nk = log_z_given_y_phi.exp()
    
    r_nk = log_z_given_y_phi.exp()
    if decoder_type == 'bernoulli':
        recon_loss = weighted_bernoulli_loglike(X, X_recon, r_nk)
    elif decoder_type == 'standard':
        mean_pred, var_pred = X_recon
        recon_loss = weighted_diag_gaussian_loglike(X, mean_pred, var_pred, r_nk)
    else:
        raise NotImplementedError
    
    # E[log q(x, z=k|y, phi)] note that this is not technically the kl-div, rather the entropy of q
    N = X_samples.shape[0]
    eta1_phi_tilde, eta2_phi_tilde = phi_tilde
    eta1_phi_tilde = eta1_phi_tilde.squeeze(-1)
    
    logprob_x_given_phi = gaussian.log_prob_nat_per_sample(X_samples, eta1_phi_tilde, eta2_phi_tilde)
    log_numerator = logprob_x_given_phi + log_z_given_y_phi.unsqueeze(-1) # (N, K, S)
    
    logprob_x_given_theta = student_t.logprob_per_sample(X_samples, mu_theta, sigma_theta, dof)
    log_denominator = logprob_x_given_theta + expected_log_pi_theta.unsqueeze(0).unsqueeze(-1)
    
    kl_div = torch.mean(torch.sum(r_nk.unsqueeze(-1) * (log_numerator - log_denominator), dim=1))
    
    elbo = -(recon_loss - kl_div)
    dbg = (recon_loss, torch.sum(r_nk * log_numerator.mean(-1)), torch.sum(r_nk * log_denominator.mean(-1)), kl_div)
    
    return elbo, dbg

def update_gmm_params(gmm_params_curr, gmm_params_new, step_size):
    updates = []
    for i, (curr_param, new_param) in enumerate(zip(gmm_params_curr, gmm_params_new)):
        updates.append((1-step_size) * curr_param + step_size * new_param)
    return updates

def unpack_recognition_gmm(phi_gmm):
    eta1, L_k_raw, pi_k_raw = phi_gmm # output from the recognition potential
    
    L_k = torch.tril(L_k_raw)
    N, D, _ = L_k.shape
    mask = torch.eye(D).unsqueeze(0).tile([N, 1, 1]).to(DEVICE)
    L_k = F.softplus(torch.diagonal(L_k, dim1=-2, dim2=-1)).unsqueeze(-1).tile([1, 1, D]) * mask + L_k * (1-mask)
    P = torch.matmul(L_k, L_k.transpose(-2, -1)) # precision
    
    eta2 = -0.5 * P
    pi_k = F.softmax(pi_k_raw, dim=0)
    return eta1, eta2, pi_k

def unpack_smm(theta_smm):
    mu, L_k_raw = theta_smm
    L_k = torch.tril(L_k_raw)
    L_k = L_k.fill_diagonal_(F.softplus(torch.diagonal(L_k, dim1=-2, dim2=-1)))
    sigma = torch.matmul(L_k, L_k.transpose(-2, -1))
    return (mu, sigma)

def compute_log_z_given_y(eta1_phi1, eta2_phi1, eta1_phi2, eta2_phi2, pi_phi2):
    N, D = eta1_phi1.shape
    assert eta2_phi1.shape == (N, D, D)
    K, D_ = eta1_phi2.shape
    assert D == D_
    assert eta2_phi2.shape == (K, D, D)
    
    eta2_phi_tilde = eta2_phi1.unsqueeze(1) + eta2_phi2.unsqueeze(0)
    
    solved = torch.linalg.solve(torch.tile(eta2_phi2.unsqueeze(0), (N, 1, 1, 1)), eta2_phi_tilde)
    w_eta2 = torch.einsum('nij,nkjl->nkil', eta2_phi1, solved)
    w_eta2 = 0.5 * (w_eta2 + w_eta2.transpose(-1, -2)) # symmetrise for numerical stability
    
    w_eta1 = torch.einsum('nij,nkuv->nkj', eta2_phi1, torch.linalg.solve(eta2_phi_tilde, torch.tile(eta1_phi2.unsqueeze(0).unsqueeze(-1), [N, 1, 1, 1])))
    
    mu_phi1, _ = gaussian.natural_to_standard(eta1_phi1, eta2_phi1)
    return gaussian.log_prob_nat(mu_phi1, w_eta1, w_eta2, pi_phi2), (w_eta1, w_eta2) # (N, K)

def sample_x_per_comp(eta1, eta2, num_samples):
    inv_sigma = -2 * eta2
    N, K, _, D = eta2.shape
    
    L = torch.linalg.cholesky(inv_sigma)
    eps = torch.randn((N, K, D, num_samples)).to(DEVICE)
    eps = torch.linalg.solve(L.transpose(-1, -2), eps)
    
    # reparameterisation sampling
    x_samples = torch.transpose(torch.linalg.solve(inv_sigma, eta1) + eps, -1, -2)
    return x_samples

def subsample_x(X_samples, log_q_z_given_y):
    N, K, S, D = X_samples.shape
    n_idx = torch.tile(torch.arange(0, N, dtype=torch.long).unsqueeze(-1), [1, S]).to(DEVICE)
    s_idx = torch.tile(torch.arange(0, S, dtype=torch.long).unsqueeze(0), [N, 1]).to(DEVICE)
    
    z_samples = torch.multinomial(torch.softmax(log_q_z_given_y, dim=-1), S, replacement=True).type(torch.long)
    choices = torch.cat([n_idx.unsqueeze(-1), z_samples.unsqueeze(-1), s_idx.unsqueeze(-1)], dim=-1)
    return gather_nd(X_samples, choices)
    

def weighted_bernoulli_loglike(X, logits_pred, r_nk=None):
    if r_nk is None:
        N, S, D = logits_pred.shape
        assert X.shape == (N, D)
    else:
        N, K, S, D = logits_pred.shape
        assert X.shape == (N, D)
        assert r_nk.shape == (N, K)
    
    X = X.unsqueeze(1)
    if r_nk is not None:
        X = X.unsqueeze(1)
    
    logprobs = -(1 + (-logits_pred * X).exp()).log().sum(-1).mean(-1)
    if r_nk is not None:
        logprobs = torch.sum(r_nk * logprobs, dim=-1)
    return logprobs.mean()

def weighted_diag_gaussian_loglike(X, mean, var, weights=None):
    if weights is None:
        mean = mean if len(mean.shape) == 3 else mean.unsqueeze(1)
        var = var if len(var.shape) == 3 else var.unsqueeze(1)
        cov_mat = torch.eye(var.shape[-1]).unsqueeze(0).unsqueeze(0) * torch.tile(var.unsqueeze(-1), [1, 1, 1, var.shape[-1]])
        logprobs = MultivariateNormal(mean, cov_mat).log_prob(X.unsqueeze(1)).sum(-1).mean() # (N, K, D) -> (N, K) -> ()
    else:
        N, K, S, D = mean.shape
        assert var.shape == (N, K, S, D)
        assert weights.shape == (N, K)
        X = X.unsqueeze(1).unsqueeze(1)
        logprobs = torch.einsum('nksd,nk->', torch.square(X-mean)/var + torch.log(var+ 1e-8), weights)
        logprobs = -0.5 * logprobs / S - N*D*torch.log(2*torch.tensor(np.pi))
        logprobs = logprobs / N
    return logprobs

def svae_main(train_loader, test_loader, latent_dim, hidden_dims, num_mixtures, act_fn, lr, lr_cvi, writer, eval_freq, 
              num_iters, num_latent_train, num_latent_test, latent_type='natparam', output_type='standard', pgm_type='gmm', 
              std_init=0.01, device=DEVICE, grad_clip=None, verbose_freq=2500, logdir=None, lr_cvi_decay=1., lr_cvi_decay_freq=1000, 
              full_dependency=False):
    X_train_sample = next(iter(train_loader))[0]
    N, D = X_train_sample.shape
    model = SVAE(D, latent_dim, enc_hidden_dims=hidden_dims, dec_hidden_dims=hidden_dims, std_init=std_init, num_mixture=num_mixtures, 
                 num_latent_sample=num_latent_train, enc_output_type=latent_type, dec_output_type=output_type, pgm_type=pgm_type, 
                 act_fn=act_fn, full_dependency=full_dependency).to(device)
    optimiser = optim.Adam(model.parameters(), lr)
    
    iter_count = 0
    num_epochs = 0
    
    while iter_count < num_iters:
        model.train()
        for (X_train, _) in train_loader:
            X_train = X_train.float().to(device)
            elbo, dbg, update_stats = model.elbo(X_train)
            optimiser.zero_grad()
            elbo.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()
            
            with torch.no_grad():
                curr_lr_cvi = cvi_lr_schedule(lr_cvi, decay_rate=lr_cvi_decay, global_step=iter_count, decay_freq=lr_cvi_decay_freq)
                X_subsamples, log_z_given_y_phi = update_stats
                res_nk = log_z_given_y_phi.exp()
                theta_star = model.m_step(X_subsamples, res_nk)
                update = model.update_gmm_params(theta_star, curr_lr_cvi)
            
            writer.add_scalar('train/elbo', elbo, iter_count)
            writer.add_scalar('train/recon-loss', dbg[0], iter_count)
            writer.add_scalar('train/kl-div', dbg[-1], iter_count)
            writer.add_scalar('train/log-numerator', dbg[1], iter_count)
            writer.add_scalar('train/log-denominator', dbg[2], iter_count)
            writer.add_scalar('train/lr-cvi', curr_lr_cvi, iter_count)
            
            # evaluate training performance in terms of mse of reconstruction
            model.eval()
            mse_train, ll_train = 0, 0
            with torch.no_grad():
                y_recon, phi_recognition, X_samples, X_subsamples, log_z_given_y_phi, phi_tilde = model(X_train)
                if output_type == 'bernoulli':
                    mse_train += weighted_mse(X_train, y_recon, log_z_given_y_phi.exp())
                    ll_train += weighted_bernoulli_loglike(X_train, y_recon, log_z_given_y_phi)
                elif output_type == 'standard':
                    mse_train += weighted_mse(X_train, y_recon[0], log_z_given_y_phi.exp())
                    ll_train += weighted_diag_gaussian_loglike(X_train, y_recon[0], y_recon[1], log_z_given_y_phi)
            
            writer.add_scalar('train-perf/mse-recon', mse_train, iter_count)
            writer.add_scalar('train-perf/loglike-recon', ll_train, iter_count)
            
            iter_count += 1
        
        num_epochs += 1            
        if (num_epochs + 1) % eval_freq == 0:
            model.eval()
            elbo_ac, recon_ac, kl_ac, log_numerator_ac, log_denominator_ac = 0, 0, 0, 0, 0
            mse_test = 0
            ll_test = 0
            with torch.no_grad():
                for (X_test, _) in test_loader:
                    X_test = X_test.float().to(device)
                    
                    y_recon, phi_recognition, X_samples, X_subsamples, log_z_given_y_phi, phi_tilde = model(X_test)
                    if output_type == 'bernoulli':
                        mse_test += weighted_mse(X_test, y_recon, log_z_given_y_phi.exp()) * X_test.shape[0]
                        ll_test += weighted_bernoulli_loglike(X_test, y_recon, log_z_given_y_phi) * X_test.shape[0]
                    elif output_type == 'standard':
                        mse_test += weighted_mse(X_test, y_recon[0], log_z_given_y_phi.exp()) * X_test.shape[0]
                        ll_test += weighted_diag_gaussian_loglike(X_test, y_recon[0], y_recon[1], log_z_given_y_phi) * X_test.shape[0]
                        
                    elbo, dbg, _ = model.elbo(X_test)
                    elbo_ac += elbo * X_test.shape[0]
                    recon_ac += dbg[0] * X_test.shape[0]
                    kl_ac += dbg[-1] * X_test.shape[0]
                    log_numerator_ac += dbg[1] * X_test.shape[0]
                    log_denominator_ac += dbg[2] * X_test.shape[0]
                    
                avg_elbo = elbo_ac / len(test_loader.dataset)
                avg_recon = recon_ac / len(test_loader.dataset)
                avg_kl = kl_ac / len(test_loader.dataset)
                avg_log_numerator = log_numerator_ac / len(test_loader.dataset)
                avg_log_denominator = log_denominator_ac / len(test_loader.dataset)
                mse_test = mse_test / len(test_loader.dataset)
                ll_test = ll_test / len(test_loader.dataset)
                
                writer.add_scalar('test/elbo', avg_elbo, iter_count)
                writer.add_scalar('test/recon-loss', avg_recon, iter_count)
                writer.add_scalar('test/kl-div', avg_kl, iter_count)
                writer.add_scalar('test/log-numerator', avg_log_numerator, iter_count)
                writer.add_scalar('test/log-denominator', avg_log_denominator, iter_count)
                writer.add_scalar('test/mse-recon', mse_test, iter_count)
                writer.add_scalar('test/loglike-recon', ll_test, iter_count)
    torch.save(model, os.path.join(logdir, 'saved_model.pt'))
        
if __name__=='__main__':
    args = parse_args()
    log_id = generate_log_id_new(args)
    logdir = os.path.join(args.logdir, log_id)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    print(logdir)
    writer = SummaryWriter(logdir)
    
    hidden_dims = [args.hidden_dim] * args.num_hidden
    torch.manual_seed(args.seed)
    train_loader, test_loader, _ = make_dataloader(args.dataset, ratio_train=args.train_ratio, ratio_val=None, 
                                                            datapath=args.datapath, batch_size=args.train_bs, test_bs=args.test_bs, n_towers=1, 
                                                            seed_split=args.seed, seed_minibatch=args.seed)
    svae_main(train_loader, test_loader, args.latent_dim, hidden_dims, args.num_mixtures, args.act_fn, args.lr, 
              args.lr_cvi, writer, eval_freq=args.eval_freq, num_iters=args.num_iters, 
              num_latent_train=args.num_latent, num_latent_test=args.num_latent, full_dependency=args.full_dependency, logdir=logdir)