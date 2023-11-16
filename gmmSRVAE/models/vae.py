from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import MultivariateNormal, Bernoulli, Normal
from torch.utils.tensorboard import SummaryWriter

from models.networks import ResMLP
from helpers.data import make_dataloader
from helpers.general_utils import generate_log_id, generate_param_schedule
from helpers.losses import weighted_mse, generate_missing_data_mask, imputation_mse
from helpers.model_utils import random_partial_isometry, View

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, enc_hidden_dims, dec_hidden_dims, std_init, 
                 num_latent_sample, seed=0, output_type='standard', act_fn='tanh'):
        super().__init__()
        self.input_dim = input_dim
        self.output_type = output_type
        self.S_latent = num_latent_sample
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.encoder = ResMLP(input_dim, enc_hidden_dims, latent_dim, std_init, act_fn=act_fn, seed=seed)
        self.decoder = ResMLP(latent_dim, dec_hidden_dims, output_dim, std_init=std_init, output_type=output_type, act_fn=act_fn, seed=seed)
        self.epsilon_distribution = Normal(torch.tensor(0.0).cuda(), torch.tensor(1.0).cuda())
    
    def forward(self, x):
        z_mean, z_var = self.encode(x)
        z_samples = self.reparameterise(z_mean, z_var) # 
        pred_x_mean, pred_x_var = self.decode(z_samples)
        return z_mean, z_var, z_samples, pred_x_mean, pred_x_var
        
    def encode(self, x):
        return self.encoder(x)
    
    
    def decode(self, z):
        out = self.decoder(z)
        if self.output_type == 'bernoulli':
            out = torch.sigmoid(out)
        return out
    
    def reparameterise(self, mean, var):
        eps = self.epsilon_distribution.sample((mean.shape[0], self.S_latent, self.latent_dim)).to(DEVICE)
        return mean.unsqueeze(1) + torch.sqrt(var).unsqueeze(1) * eps
    
    def elbo(self, x):
        batch_size = x.shape[0] # (N, D)
        z_mean, z_var, z_samples, pred_x_mean, pred_x_var = self.forward(x) # pred_x_mean: (N, S, D)
        if self.output_type == 'bernoulli':
            recon_loss = Bernoulli(pred_x_mean).log_prob(x.unsqueeze(1)).sum(-1).mean()
        else:
            sample_mean = torch.sum(torch.square(x.unsqueeze(1)-pred_x_mean)/pred_x_var) + torch.sum(torch.log(pred_x_var))
            sample_mean = sample_mean / self.S_latent
            loglike = -0.5 * (sample_mean + batch_size * self.latent_dim*torch.log(2.*torch.tensor(np.pi)))
            recon_loss = loglike / batch_size
            # recon_loss = MultivariateNormal(pred_x_mean, pred_x_var.unsqueeze(-1).repeat(1, 1, 1, self.input_dim)\
                # *torch.eye(self.input_dim).unsqueeze(0).unsqueeze(0).repeat(batch_size, self.S_latent, 1, 1).to(DEVICE)).log_prob(x.unsqueeze(1)).sum(-1).mean(-1).mean(-1)
        kl_div = 0.5 * torch.mean(torch.sum(-1-z_var.log() + z_var + torch.square(z_mean), dim=1))
        elbo = recon_loss - kl_div
        return -elbo, recon_loss, kl_div
    
def vae_main(train_loader, test_loader, latent_dim, hidden_dims, act_fn, lr, writer, eval_freq, 
             num_epochs=100, device='cuda:0', grad_clip=None, logdir=None, verbose_freq=2500):
    X_train_sample = next(iter(train_loader))[0]
    N, D = X_train_sample.shape
    model = VAE(D, latent_dim, D, hidden_dims, hidden_dims, std_init=.01, num_latent_sample=10, seed=0, act_fn=act_fn).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    
    iter_count = 0
    num_epochs = 0
    while iter_count < 120000:
        model.train()
        for (X_train, _) in train_loader:
            X_train = X_train.float().to(device)
            elbo, recon_loss, kl_div = model.elbo(X_train)
            optimiser.zero_grad()
            elbo.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()

            writer.add_scalar('train/elbo', elbo, iter_count)
            writer.add_scalar('train/recon-loss', recon_loss, iter_count)
            writer.add_scalar('train/kl-div', kl_div, iter_count)
            
            with torch.no_grad():
                z_mean, z_var, z_samples, x_pred_mean, x_pred_var = model(X_train)
                weights = torch.ones((X_train.shape[0], 1)).to(DEVICE)
                x_pred_mean = x_pred_mean.unsqueeze(1)
                mse_train = weighted_mse(X_train, x_pred_mean, weights)
            writer.add_scalar('train/mse-recon', mse_train, iter_count)
            
            iter_count += 1
            
        
        num_epochs += 1
        
        if (num_epochs+1) % eval_freq == 0:
            model.eval()
            elbo_ac, recon_ac, kl_ac = 0, 0, 0
            mse_test = 0
            with torch.no_grad():
                for (X_test, _) in test_loader:
                    X_test = X_test.float().to(device)
                    elbo, recon_loss, kl_div = model.elbo(X_test)
                    elbo_ac += elbo * X_test.shape[0]
                    recon_ac += recon_loss * X_test.shape[0]
                    kl_ac = kl_div * X_test.shape[0]
                    
                    z_mean, z_var, z_samples, x_pred_mean, x_pred_var = model(X_test)
                    weights = torch.ones((X_test.shape[0], 1)).to(DEVICE)
                    x_pred_mean = x_pred_mean.unsqueeze(1)
                    mse_test += weighted_mse(X_test, x_pred_mean, weights) * X_test.shape[0]
                    
                avg_elbo = elbo_ac / len(test_loader.dataset)
                avg_recon = recon_ac / len(test_loader.dataset)
                avg_kl = kl_ac / len(test_loader.dataset)
                mse_test = mse_test / len(test_loader.dataset)
            writer.add_scalar('test/elbo', avg_elbo, iter_count)
            writer.add_scalar('test/recon-loss', avg_recon, iter_count)
            writer.add_scalar('test/kl-div', avg_kl, iter_count)
            writer.add_scalar('test/mse-pred', mse_test, iter_count)
        
        if (iter_count+1) % verbose_freq == 0:
            print(f'epoch: {num_epochs+1:d} | elbo: {elbo:.2f} | recon-loss: {recon_loss:.2f} | kl-div: {kl_div:.2f}')
            
    if logdir is not None:
        torch.save(model, os.path.join(logdir, 'vae_model.pt'))

if __name__=='__main__':
    datapath = './datasets'
    ratio_train = 0.7
    ratio_val = None
    ratio_test = 0.3
    num_iter = 1000
    eval_freq = 10
    K = 10
    seed = 1
    seed_data = 0
    logdir = './logdir'
    
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--seed', default=0, type=int)
    args = args.parse_args()

    schedule = generate_param_schedule({
        'method': 'vae', 
        'dataset': 'pinwheel', 
        'lr': 3e-4,
        'U': 50,
        'L': 6, 
        'H': 2, 
        'seed': args.seed,
        'act_fn': 'tanh', 
    })
    
    for config_id, config in enumerate(schedule):
        lr = config['lr']
        hidden_dims = [config['U']] * config['H']
        latent_dim = config['L']
        seed = config['seed']
        dataset = config['dataset']
        act_fn = 'tanh'
        print(f'Experiment {config_id} with config: \n{config}')
        
        log_id = generate_log_id(config)
        logdir = os.path.join(logdir, log_id)
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        print(logdir)
        writer = SummaryWriter(logdir)
        
        torch.manual_seed(seed)
        train_loader, test_loader, _ = make_dataloader(dataset, ratio_train=ratio_train, ratio_val=None, 
                                                                datapath=datapath, batch_size=64, test_bs=100, n_towers=1, 
                                                                seed_split=seed_data, seed_minibatch=seed_data)
        # train_loader = iter(train_loader)
        # test_loader = iter(test_loader)
        # X_train, y_train = next(train_loader)
        # X_test, y_test = next(test_loader)
        
        # i, r_nk, log_r_nk, theta, stats = variational_EM_smm(X_train, X_test, writer, kappa_init, K=K, num_iter=num_iter, threshold=0.001, 
        #                                                      seed=seed, eval_freq=eval_freq)
        vae_main(train_loader, test_loader, latent_dim, hidden_dims, act_fn, lr, writer, eval_freq=1, num_epochs=15, logdir=logdir, device='cuda:0')