import argparse
import io
import os
import copy
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions import RelaxedBernoulli
from torch.utils.tensorboard import SummaryWriter
import PIL
import pickle

from experiments.data_experiment import make_dataloader
from distributions import bernoulli
from factorgraph import Graph
from factorgraph_numpy import Graph_numpy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class binTreeSRVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, prior_w=None, recognition_MF=True, latent_pgm='MF'):
        super(binTreeSRVAE, self).__init__()
        
        self.prior_w = prior_w # (latent_dim-1, 2)
        self.latent_pgm = latent_pgm
        self.latent_dim = latent_dim
        self.recognition_MF = recognition_MF
        if prior_w is not None:
            assert prior_w.shape[0] == latent_dim-1, 'the dimension of the input prior does not match the latent dimension'

        if latent_pgm == 'MF':
            self.prior_logits = bernoulli.standard_to_natural(torch.ones((latent_dim, )) * 0.5).to(device)
            self.prior_p = 0.5
        elif latent_pgm == 'tree':
            self.prior_pgm = self.init_tree_prior()
            iters, converged = self.prior_pgm.lbp(normalize=True)
        else:
            raise NotImplementedError
        
        enc_dims = [input_dim] + hidden_dims
        encoder_layers = []
        for d1, d2 in zip(enc_dims[:-1], enc_dims[1:]):
            encoder_layers.extend([nn.Linear(d1, d2), nn.ReLU()])
        if recognition_MF:
            encoder_layers.extend([nn.Linear(enc_dims[-1], latent_dim)])
        else:
            encoder_layers.extend([nn.Linear(enc_dims[-1], latent_dim+2*(latent_dim-1))])
        self.encoder = nn.Sequential(*encoder_layers)
        
        dec_dims = [latent_dim] + list(reversed(hidden_dims))
        decoder_layers = []
        for d1, d2 in zip(dec_dims[:-1], dec_dims[1:]):
            decoder_layers.extend([nn.Linear(d1, d2), nn.ReLU()])
        decoder_layers.extend([nn.Linear(dec_dims[-1], input_dim)])
        self.decoder = nn.Sequential(*decoder_layers)
        
    def init_tree_prior(self):
        prior_pgm = Graph(debug=False)
        for i in range(self.latent_dim):
            prior_pgm.rv(f'v_{i}', n_opts=2)
            if i == 0:
                prior_pgm.factor([f'v_{i}'], potential=torch.FloatTensor([0.5, 0.5])) # need to change all torch.FloatTensor to torch.FloatTensor if using cuda
            else:
                prior_pgm.factor([f'v_{i}'], potential=torch.FloatTensor([1.0, 1.0]))
        prior_pgm.factor(['v_0', 'v_1'], potential=torch.FloatTensor([[0.5, 0.5], [0.5, 0.5]]) if self.prior_w is None else 
                         torch.FloatTensor([[F.sigmoid(self.prior_w[0, 0]), 1-F.sigmoid(self.prior_w[0, 0])], 
                                            [F.sigmoid(self.prior_w[0, 1]), 1-F.sigmoid(self.prior_w[0, 1])]]))
        count = 1
        for i in range(int(np.log2(self.latent_dim-1))):
            for j in range(int(2**i)):
                prior_pgm.factor([f'v_{int(2**i+j)}', f'v_{int((2**i+j)*2)}'], 
                                 potential=torch.FloatTensor([[0.5, 0.5], [0.5, 0.5]]) if self.prior_w is None else
                                 torch.FloatTensor([[F.sigmoid(self.prior_w[count, 0]), 1-F.sigmoid(self.piror_w[count, 0])], 
                                                    [F.sigmoid(self.prior_w[count, 1]), 1-F.sigmoid(self.prior_w[count, 1])]]))
                prior_pgm.factor([f'v_{int(2**i+j)}', f'v_{int((2**i+j)*2+1)}'], 
                                 potential=torch.FloatTensor([[0.5, 0.5], [0.5, 0.5]]) if self.prior_w is None else 
                                 torch.FloatTensor([[F.sigmoid(self.prior_w[count+1, 0]), 1-F.sigmoid(self.piror_w[count+1, 0])], 
                                                    [F.sigmoid(self.prior_w[count+1, 1]), 1-F.sigmoid(self.prior_w[count+1, 1])]]))
                count = count + 2
        return prior_pgm
        
    def forward(self, x, temp=1.0, hard=False):
        batch_size = x.shape[0]
        logits = self.encode(x)
        
        if self.latent_pgm == 'MF':
            e_step_probs = self.e_step(logits)
            q_z = RelaxedBernoulli(temp, probs=e_step_probs)
            probs = q_z.probs
            z_samples = q_z.rsample()
            prob_terms = (probs)
            if hard:
                z_samples = self.hard_sample(z_samples)
        elif self.latent_pgm == 'tree':
            amortised_pgm = self.e_step(logits)
            amortised_node_marg, amortised_pairwise_marg = self.extract_tree_marginals(amortised_pgm, batch_size)
            z_samples = self.tree_sampling(amortised_node_marg, amortised_pairwise_marg, batch_size, temp=temp, hard=hard)
            prob_terms = (amortised_node_marg, amortised_pairwise_marg)

        recon_logits = self.decode(z_samples)
        return recon_logits, prob_terms
    
    def extract_tree_marginals(self, tree_pgm, normalize=True):
        _, node_marginals = tree_pgm.rv_marginals(normalize=normalize)
        pairwise_marginals = tree_pgm.pairwise_marginals(normalize=normalize)
        return node_marginals, pairwise_marginals
    
    def hard_sample(self, z):
        z_hard = 0.5 * (torch.sign(z - 0.5) + 1)
        return z + (z_hard-z).detach()
    
    def tree_sampling(self, node_marginals, pairwise_marginals, batch_size, temp, hard=False):
        sample = torch.zeros((batch_size, self.latent_dim)).to(device)
        q_z_0 = RelaxedBernoulli(temp, probs=node_marginals['v_0'][:, 1])
        s = q_z_0.rsample()
        if hard: 
            s = self.hard_sample(s)
            sample[:, 0] = s.clone()
            next_ind = s.long().clone()
        else:
            sample[:, 0] = s.clone()
            with torch.no_grad():
                next_ind = self.hard_sample(s).long()
        pairwise_probs = (pairwise_marginals['f(v_0, v_1)'] / (pairwise_marginals['f(v_0, v_1)'].sum(dim=-1, keepdim=True)+1e-10))[:, :, 1]
        q_z_1_init = RelaxedBernoulli(temp, probs=pairwise_probs[torch.arange(pairwise_probs.shape[0]), next_ind])
        s = q_z_1_init.rsample()
        if hard:
            s = self.hard_sample(s)
            sample[:, 1] = s.clone()
            next_ind = s.long().clone()
        else:
            sample[:, 1] = s.clone()
            with torch.no_grad():
                next_ind = self.hard_sample(s).long()
        with torch.no_grad():
            next_ind_list = [next_ind]
        for i in range(int(np.log2(self.latent_dim-1))):
            next_ind_list_new = []
            for j in range(int(2**i)):
                pairwise_probs_1 = (pairwise_marginals[f'f(v_{int(2**i+j)}, v_{int((2**i+j)*2)})'] / \
                    (pairwise_marginals[f'f(v_{int(2**i+j)}, v_{int((2**i+j)*2)})'].sum(dim=-1, keepdim=True)+1e-10))[:, :, 1]
                q_z_1 = RelaxedBernoulli(temp, probs=pairwise_probs_1[torch.arange(pairwise_probs_1.shape[0]), next_ind_list[j]])
                s_1 = q_z_1.rsample()
                if hard:
                    s_1 = self.hard_sample(s_1)
                    sample[:, int((2**i+j)*2)] = s_1.clone()
                    next_ind = s_1.long().clone()
                else:
                    sample[:, int((2**i+j)*2)] = s_1.clone()
                    with torch.no_grad():
                        next_ind = self.hard_sample(s_1).long()
                with torch.no_grad():
                    next_ind_list_new.append(next_ind)
                pairwise_probs_2 = (pairwise_marginals[f'f(v_{int(2**i+j)}, v_{int((2**i+j)*2+1)})'] / \
                    (pairwise_marginals[f'f(v_{int(2**i+j)}, v_{int((2**i+j)*2+1)})'].sum(dim=-1, keepdim=True)+1e-10))[:, :, 1]
                q_z_2  = RelaxedBernoulli(temp, probs=pairwise_probs_2[torch.arange(pairwise_probs_2.shape[0]), next_ind_list[j]])
                s_2 = q_z_2.rsample()
                if hard:
                    s_2 = self.hard_sample(s_2)
                    sample[:, int((2**i+j)*2+1)] = s_2.clone()
                    next_ind = s_2.long().clone()
                else:
                    sample[:, int((2**i+j)*2+1)] = s_2.clone()
                    with torch.no_grad():
                        next_ind = self.hard_sample(s_2).long()
                with torch.no_grad():
                    next_ind_list_new.append(next_ind)
            next_ind_list = copy.deepcopy(next_ind_list_new)
        return sample
        
    def encode(self, x):
        out = self.encoder(x)
        return out
        
    def decode(self, z):
        out = self.decoder(z)
        return F.sigmoid(out)
    
    def e_step(self, logits):
        if self.latent_pgm == 'MF':
            out_probs = F.sigmoid(logits)
            return out_probs
        elif self.latent_pgm == 'tree':
            amortised_pgm = copy.deepcopy(self.prior_pgm)
            if self.recognition_MF:
                # recognition network only output singleton potentials
                for i in range(self.latent_dim):
                    curr_potential = amortised_pgm._factors[i].get_potential().clone()
                    amortised_pgm._factors[i].set_potential(curr_potential * 
                                                            torch.cat([F.sigmoid(logits[:, [i]]), 1-F.sigmoid(logits[:, [i]])], dim=-1))
            else:
                singleton_logits, pairwise_logits_1, pairwise_logits_2 = torch.split(logits, (self.latent_dim, self.latent_dim-1, self.latent_dim-1), dim=-1)
                for i in range(self.latent_dim):
                    curr_potential = amortised_pgm._factors[i].get_potential().clone()
                    amortised_pgm._factors[i].set_potential(curr_potential * 
                                                            torch.cat([F.sigmoid(singleton_logits[:, [i]]), 1-F.sigmoid(singleton_logits[:, [i]])], dim=-1))
                for i in range(int(self.latent_dim-1)):
                    curr_potential = amortised_pgm._factors[i+self.latent_dim].get_potential()
                    amortised_pgm._factors[i+self.latent_dim].set_potential(curr_potential * \
                        torch.cat([torch.cat([F.sigmoid(pairwise_logits_1[:, [i]]) * F.sigmoid(pairwise_logits_2[:, [i]]), 
                                              F.sigmoid(pairwise_logits_1[:, [i]]) * (1-F.sigmoid(pairwise_logits_2[:, [i]]))], dim=-1).unsqueeze(1), 
                                   torch.cat([(1-F.sigmoid(pairwise_logits_1[:, [i]])) * F.sigmoid(pairwise_logits_2[:, [i]]), 
                                              (1-F.sigmoid(pairwise_logits_1[:, [i]])) * (1-F.sigmoid(pairwise_logits_2[:, [i]]))], dim=-1).unsqueeze(1)], 
                                  dim=1).to(device))
            
            amortised_pgm.lbp(normalize=True)
            return amortised_pgm
        else:
            raise NotImplementedError
    
    def free_energy(self, x, temp=1.0, hard=False):
        recon_logits, prob_terms = self(x, temp, hard)
        recon_loss = F.binary_cross_entropy(recon_logits, x, reduction='sum')
        if self.latent_pgm == 'MF':
            latent_p = prob_terms
            kl_div = (latent_p * ((latent_p + 1e-10)/self.prior_p).log() + (1-latent_p) * ((1-latent_p+1e-10)/self.prior_p).log()).sum()
        elif self.latent_pgm == 'tree':
            amortised_node_marg, amortised_pairwise_marg = prob_terms
            with torch.no_grad():
                _, prior_node_marg = self.prior_pgm.rv_marginals(normalize=True)
                prior_pairwise_marg = self.prior_pgm.pairwise_marginals(normalize=True)
            singleton_kl, pairwise_kl = 0.0, 0.0
            assert set(amortised_node_marg.keys()) == set(prior_node_marg.keys()) and \
                set(amortised_pairwise_marg.keys()) == set(prior_pairwise_marg.keys()), 'prior and variational distributions have different nodes/edges'
            for p in amortised_pairwise_marg.keys():
                pairwise_kl = pairwise_kl + torch.sum(amortised_pairwise_marg[p] * ((amortised_pairwise_marg[p]+1e-10).log() - 
                                                                                    (prior_pairwise_marg[p]+1e-10).log()))
            for p in amortised_node_marg.keys():
                singleton_kl = singleton_kl + torch.sum((self.prior_pgm._rvs[p].n_neighbours()-1) * amortised_node_marg[p] * 
                                                        ((amortised_node_marg[p]+1e-10).log() - (prior_node_marg[p]+1e-10).log()))
            kl_div = pairwise_kl - singleton_kl
        return recon_loss + kl_div, recon_loss, kl_div

def natural_to_standard(eta):
    return 1./(1+torch.exp(-eta))

def main(input_dim, train_loader, test_loader, latent_dim, hidden_dims, temp_init, temp_anneal_rate, temp_anneal_freq, temp_min, 
         device, writer, num_epoch=10, lr=1e-3, grad_clip=None, log_freq=10, hard=False, logdir=None, dataset='bar-test', 
         latent_pgm='tree', recognition_MF=False, sampling_temperature=None):
    model = binTreeSRVAE(input_dim, latent_dim, hidden_dims, prior_w=None, recognition_MF=recognition_MF, latent_pgm=latent_pgm).to(device)
    optimiser = optim.Adam(model.parameters(), lr)
    
    temp = temp_init
    
    train_logs, test_logs = [], []
    
    iter_count = 0
    for e in range(num_epoch):
        model.train()
        train_losses = 0.
        for batch_idx, train_data in enumerate(train_loader):
            if dataset == 'mnist':
                X_train, _ = train_data
            elif dataset == 'bar-test':
                X_train = train_data[0]
            elif dataset == 'video_vae':
                X_train = train_data[0]
            X_train = X_train.float().reshape(X_train.shape[0], -1).to(device)
            # writer.add_graph(model, X_train)
            optimiser.zero_grad()
            with torch.cuda.amp.autocast(enabled=False):
                free_energy, recon_loss, kl_div = model.free_energy(X_train, temp, hard)
            free_energy.backward(retain_graph=True)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()
            
            train_losses = train_losses + free_energy.item()
            if (batch_idx) % log_freq == 0:
                print(f'Train Epoch: {e} | {batch_idx*X_train.shape[0]}/Progress: {len(train_loader.dataset)} ({100.*batch_idx/len(train_loader):.2f}% | \
                    Free Energy: {free_energy.item():.4f}')
            
            writer.add_scalar('train/free_energy', free_energy.item()/X_train.shape[0], iter_count)
            writer.add_scalar('train/recon_loss', recon_loss.item()/X_train.shape[0], iter_count)
            writer.add_scalar('train/kl_div', kl_div.item()/X_train.shape[0], iter_count)
            
            with torch.no_grad():
                train_logs.append([iter_count, free_energy.item()/X_train.shape[0], 
                                   recon_loss.item()/X_train.shape[0], kl_div.item()/X_train.shape[0]])
            iter_count = iter_count + 1
            
            if iter_count % temp_anneal_freq == 0:
                temp = max(temp * np.exp(-temp_anneal_rate * iter_count), temp_min)
        
        print(f'--------------------------------------------\n \
            Epoch: {e} | Average Free Energy: {train_losses/len(train_loader.dataset):.4f}\n\
            -----------------------------------------------')
        
        # testing...
        model.eval()
        test_losses = 0.
        with torch.no_grad():
            for batch_idx, test_data in enumerate(test_loader):
                if dataset == 'mnist':
                    X_test, _ = test_data
                elif dataset == 'bar-test':
                    X_test = test_data[0]
                elif dataset == 'video_vae':
                    X_test = test_data[0]
                X_test = X_test.float().reshape(X_test.shape[0], -1).to(device)
                logits_recon, latent_p = model(X_test, temp, hard)
                free_energy, recon_loss, kl_div = model.free_energy(X_test, temp, hard)
                test_losses = test_losses + free_energy.item()
                if batch_idx == 0:
                    n = min(X_train.shape[0], 8)
                    if dataset == 'mnist':
                        comparison = torch.cat([X_test[:n].reshape(n, 1, 28, 28), logits_recon.view(args.batch_size, 1, 28, 28)[:n]])
                    elif dataset == 'bar-test':
                        comparison = torch.cat([X_test[:n].reshape(n, 1, 8, 8), logits_recon.view(args.batch_size, 1, 8, 8)[:n]])
                    elif dataset == 'video_vae':
                        comparison = torch.cat([X_test[:n].reshape(n, 1, 80, 120), logits_recon.view(args.batch_size, 1, 80, 120)[:n]])
                    else:
                        raise NotImplementedError
                    if dataset == 'bar-test':
                        if (e) % 1 == 0:
                            save_image(comparison.cpu(), os.path.join(logdir, f'recon_{e}.png'), nrow=n)
                            image = PIL.Image.open(os.path.join(logdir, f'recon_{e}.png'))
                            image = transforms.ToTensor()(image)
                            writer.add_image('recon-images', image, e)
                            os.remove(os.path.join(logdir, f'recon_{e}.png'))
                    elif dataset == 'mnist':
                        save_image(comparison.cpu(), os.path.join(logdir, f'recon_{e}.png'), nrow=n)
                        image = PIL.Image.open(os.path.join(logdir, f'recon_{e}.png'))
                        image = transforms.ToTensor()(image)
                        writer.add_image('recon-images', image, e)
                    elif dataset == 'video_vae':
                        save_image(comparison.cpu(), os.path.join(logdir, f'recon_{e}.png'), nrow=n)
                        image = PIL.Image.open(os.path.join(logdir, f'recon_{e}.png'))
                        image = transforms.ToTensor()(image)
                        writer.add_image('recon-images', image, e)
                    
                writer.add_scalar('test/free_energy', free_energy.item()/X_test.shape[0], iter_count)
                writer.add_scalar('test/recon_loss', recon_loss.item()/X_test.shape[0], iter_count)
                writer.add_scalar('test/kl_div', kl_div.item()/X_test.shape[0], iter_count)
                with torch.no_grad():
                    test_logs.append([iter_count, free_energy.item()/X_test.shape[0], 
                                    recon_loss.item()/X_test.shape[0], kl_div.item()/X_test.shape[0]])
    torch.save(model, os.path.join(logdir, 'trained_model.pt'))
    with open(os.path.join(logdir, 'train_logs.pkl'), 'wb') as f:
        pickle.dump(train_logs, f)
    f.close()
    with open(os.path.join(logdir, 'test_logs.pkl'), 'wb') as f:
        pickle.dump(test_logs, f)
    f.close()
                    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Bar Model')
    parser.add_argument('--dataset', type=str, default='bar-test', metavar='DS', 
                        help='name of the dataset to use')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--latent-dim', type=int, default=16, metavar='D', 
                        help='dimension of the latent code')
    parser.add_argument('--hidden-dim', type=int, default=256, metavar='H', 
                        help='number of hidden units in each hidden layer')
    parser.add_argument('--num-hidden', type=int, default=2, metavar='NH', 
                        help='number of hidden layers')
    parser.add_argument('--hard', type=bool, default=False, metavar='AR',
                        help='hard discretisation in sampling')
    parser.add_argument('--test-concrete', type=bool, default=False, metavar='TC',
                        help='test self-defined concrete reparameterisation')
    parser.add_argument('--grad-clip', type=float, default=0.0, 
                        help='gradient norm clipping for more stable training')
    parser.add_argument('--latent-pgm', type=str, default='tree', 
                        help='latent pgm structure (tree or mean-field)')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='learning rate of the Adam optimiser')
    parser.add_argument('--recognition-MF', type=bool, default=False, 
                        help='MF- or tree-based recognition potential')
    parser.add_argument('--tree-structured-gen', type=bool, default=False, 
                        help='whether or not to use tree-structured true generative sampling process')
    parser.add_argument('--sampling-temperature', type=float, default=10, 
                        help='temperature for true sampling process')
    parser.add_argument('--num-samples', type=int, default=5000, 
                        help='number of generated samples')
    parser.add_argument('--dim-size', type=int, default=8, 
                        help='length of each side of the bar-test observation')
    parser.add_argument('--init-prob', type=float, default=0.4, metavar='G',  
                        help='initial probability for being active for each digit in the true generative process')
    parser.add_argument('--obs-dim', type=tuple, default=(80, 120), metavar='G',  
                        help='dimension of each observation frame')
    parser.add_argument('--num-latent', type=int, default=10, metavar='G', 
                        help='number of latent chains the FHMM generative model')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device('cuda' if args.cuda else 'cpu')

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'bar-test':
        if args.tree_structured_gen:
            g = Graph_numpy()
            for i in range(16):
                g.rv(f'v_{i}', 2)
            for i in range(16):
                if i == 0:
                    g.factor([f'v_{i}'], potential=np.array([0.5, 0.5]))
                else:
                    g.factor([f'v_{i}'], potential=np.array([1.0, 1.0]))
                    
            g.factor(['v_0', 'v_1'], potential=np.array([
                [0.9, 0.1], 
                [0.8, 0.2]
            ]))
            g.factor(['v_1', 'v_2'], potential=np.array([
                [0.9, 0.1], 
                [0.7, 0.3]
            ]))
            g.factor(['v_1', 'v_3'], potential=np.array([
                [0.9, 0.1], 
                [0.7, 0.3]
            ]))
            for i in range(1, 3):
                for j in range(int(2**i)):
                    g.factor([f'v_{int(2**i+j)}', f'v_{int((2**i+j)*2)}'], potential=np.array([
                        [0.9, 0.1], 
                        [0.7, 0.3]
                    ]))
                    g.factor([f'v_{int(2**i+j)}', f'v_{int((2**i+j)*2+1)}'], potential=np.array([
                        [0.9, 0.1], 
                        [0.7, 0.3]
                    ]))
            g.lbp(normalize=True)
            permutation = {0: 0, 1: 9, 2: 10, 3: 12, 4: 2, 5: 11, 6: 4, 7: 13, 8: 3, 9: 7, 10: 6, 11: 14, 12: 15, 13: 8, 14: 5, 15: 15}
            prior_tree = g
        else:
            prior_tree = None
            permutation = None
        train_loader, test_loader, _ = make_dataloader(args.dataset, ratio_train=0.9, batch_size=args.batch_size, b=args.sampling_temperature, 
                                                       num_samples=args.num_samples, prior_tree=prior_tree, permutation=permutation, grid_dim=8, 
                                                       seed_generation=args.seed, **kwargs)
    else:
        raise NotImplementedError

    INITIAL_TEMP = 1.0
    ANNEAL_RATE = 0.00003
    MIN_TEMP = 0.1

    temp = INITIAL_TEMP
    
    hidden_dims = [args.hidden_dim] * args.num_hidden
    
    amortiser_handle = 'MF' if args.recognition_MF else 'Tree'
    sampling_handle = 'tree' if args.tree_structured_gen else 'uniform'
    if args.recognition_MF:
        if args.latent_pgm == 'MF':
            identifier = 'VAE'
        elif args.latent_pgm == 'tree':
            identifier = 'SVAE'
    else:
        if args.latent_pgm == 'tree':
            identifier = 'TreeSVAE'
        else:
            raise NotImplementedError
    if args.dataset == 'bar-test':
        logdir = os.path.join('logdir', 'official_runs', str(args.sampling_temperature), f'{args.dataset}_{sampling_handle}', identifier, str(args.seed))
    else:
        identifier = f'NL{args.num_latent}_LD{args.latent_dim}'
        logdir = os.path.join('logdir', 'official_runs', f'{args.dataset}', identifier, str(args.seed))
    print(logdir)
    writer = SummaryWriter(logdir)
    
    if args.dataset == 'bar-test':
        main(args.dim_size**2, train_loader, test_loader, latent_dim=args.latent_dim, hidden_dims=hidden_dims, temp_init=temp, 
            temp_anneal_rate=ANNEAL_RATE, temp_anneal_freq=1000, temp_min=MIN_TEMP, device=device, writer=writer, 
            hard=args.hard, logdir=logdir, num_epoch=args.epochs, latent_pgm=args.latent_pgm, recognition_MF=args.recognition_MF, 
            dataset=args.dataset)
    else:
        raise NotImplementedError