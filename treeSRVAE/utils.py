from math import log
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Uniform
import torchvision
from torchvision import datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
import io
import copy
import PIL.Image
from torchvision.transforms import ToTensor
from itertools import product
from torch.autograd import Variable

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = torch.tensor(1e-10).to(DEVICE)

def make_dataloader(dataset, ratio_train=None, ratio_val=None, binary=False, datapath='./datasets', 
                   batch_size=128, test_bs=-1, n_towers=1, n_threads=0, seed_split=0, seed_minibatch=0, 
                   dtype=torch.float32, noise_level=0.1):

    train_dataset = datasets.MNIST(f'{datapath}/mnist/train', train=True, download=True,  
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))]),
                                target_transform=torchvision.transforms.Compose([
                                    lambda x:torch.LongTensor([x]), 
                                    lambda x:F.one_hot(x,10)]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size if batch_size > 0 else train_dataset.__len__(), shuffle=True)
    test_dataset = torchvision.datasets.MNIST(f'{datapath}/mnist/test', train=False, download=True, 
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(), 
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))]), 
                                                target_transform=torchvision.transforms.Compose([
                                                    lambda x:torch.LongTensor([x]), 
                                                    lambda x:F.one_hot(x,10)]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_bs if test_bs > 0 else test_dataset.__len__(), shuffle=False)
    return train_loader, test_loader, 

def get_one_hot_vector(idx, dim=10):
    one_hot = np.zeros(dim)
    one_hot[idx] = 1.
    return one_hot

def plot_digit_grid(model, fig_size=10, digit_size=28, std_dev=2., filename='vae'):
    figure = np.zeros((digit_size * fig_size, digit_size * fig_size))
    grid_x = np.linspace(-std_dev, std_dev, fig_size)
    grid_y = np.linspace(-std_dev, std_dev, fig_size)

    model.eval()
    
    for i, xi in enumerate(grid_x):
        for j, yi in enumerate(grid_y):
            if model.latent_dim > 0:
                if model.latent_discrete_dim > 0:
                    z_sample = std_dev * np.random.rand(model.latent_dim)
                    c_sample = get_one_hot_vector(j % model.latent_discrete_dim, model.latent_discrete_dim)
                    latent_sample = torch.FloatTensor(np.hstack((z_sample, c_sample))).to(DEVICE).unsqueeze(0)
                else:
                    latent_sample = torch.FloatTensor(std_dev * np.random.rand(model.latent_dim)).to(DEVICE).unsqueeze(0)
            else:
                latent_sample = torch.FloatTensor(get_one_hot_vector(j % model.latent_discrete_dim, 
                                                                     model.latent_discrete_dim)).to(DEVICE).unsqueeze(0)
            generated = model.decode(latent_sample)
            digit = generated[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit.cpu().numpy()

    plt.imshow(figure, cmap='viridis')
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    return image

def kl_gaussian(z_mean, z_log_var):
    kl_div = .5 * torch.sum(torch.square(z_mean) + z_log_var.exp() - 1 - z_log_var, dim=-1)
    return kl_div.mean()

def kl_discrete(probs):
    N, D = probs.shape
    neg_ent = torch.sum(probs * torch.log(probs + EPS), dim=-1)
    return torch.log(torch.tensor(D)) + torch.mean(neg_ent) 

def sampling_concrete(alpha, output_shape, temperature=0.67):
    eps = torch.rand(output_shape).to(DEVICE)
    gumbel = -torch.log(-torch.log(eps+EPS)+EPS)
    logit = (torch.log(alpha + EPS) + gumbel) / temperature
    return torch.softmax(logit, dim=-1)

def sampling_normal(z_mean, z_log_var, output_shape):
    eps = torch.randn(output_shape).to(DEVICE)
    return z_mean + (z_log_var/2).exp() * eps

def sigmoid(x):
  return 1 / (1 + np.exp(-1. * x))

def to_multi_hot(dim, idx):
    N = idx.shape[0]
    assert idx.shape[1] == 2, f'the index vector must be (N, 2)-dimensional, but got {idx.shape[1]} for its second shape'
    out = np.zeros((N, dim))
    out[np.arange(N), idx[:, 0]] += 1
    out[np.arange(N), idx[:, 1]] += 1
    return out

def sample_tree(tree, permutation, num_samples):
    _, node_marginals = tree.rv_marginals(normalize=True)
    pairwise_marginals = tree.pairwise_marginals(normalize=True)
    
    D = len(tree._rvs)
    samples = np.zeros((num_samples, D))
    mapping = get_mapping(permutation)
    for k in range(num_samples):
        sample_i = np.zeros((D, ))
        sample_i[0] = np.random.binomial(n=1, p=node_marginals['v_0'][1])
        p_z0_z1 = (pairwise_marginals['f(v_0, v_1)'] / pairwise_marginals['f(v_0, v_1)'].sum(axis=-1, keepdims=True))[int(sample_i[0]), 1]
        sample_i[1] = np.random.binomial(n=1, p=p_z0_z1)
        next_ind = [sample_i[1]]
        for i in range(int(np.log2(D-1))):
            next_ind_new = []
            for j in range(int(2**i)):
                p_z0_z1 = (pairwise_marginals[f'f(v_{int(2**i+j)}, v_{int((2**i+j)*2)})'] / \
                    pairwise_marginals[f'f(v_{int(2**i+j)}, v_{int((2**i+j)*2)})'].sum(axis=-1, keepdims=True))[int(next_ind[j]), 1]
                sample_ij = np.random.binomial(n=1, p=p_z0_z1)
                sample_i[int((2**i+j)*2)] = sample_ij
                next_ind_new.append(sample_ij)
                
                p_z0_z2 = (pairwise_marginals[f'f(v_{int(2**i+j)}, v_{int((2**i+j)*2+1)})'] / \
                    pairwise_marginals[f'f(v_{int(2**i+j)}, v_{int((2**i+j)*2+1)})'].sum(axis=-1, keepdims=True))[int(next_ind[j]), 1]
                sample_ij = np.random.binomial(n=1, p=p_z0_z2)
                sample_i[int((2**i+j)*2+1)] = sample_ij
                next_ind_new.append(sample_ij)
            next_ind = copy.deepcopy(next_ind_new)
        sample_i = sample_i[mapping]
        samples[k, :] = sample_i
    return samples

def get_mapping(perm):
    out = np.zeros((len(perm), ), dtype=int)
    for i in range(len(perm)):
        out[perm[i]] = i
    return out

def get_adjacent_inds(dim):
    num_nodes = dim ** 2
    neighbours = {}
    neighbours_list = []
    for i in range(num_nodes):
        n_i = []
        n_i_pair = []
        if i + dim < num_nodes:
            n_i.append(i+dim)
            if not (i+dim, i) in neighbours_list:
                n_i_pair.append((i, i+dim))
        if i - dim >= 0:
            n_i.append(i-dim)
            if not (i-dim, i) in neighbours_list:
                n_i_pair.append((i, i-dim))
        try:
            if np.unravel_index(i-1, (dim, dim))[0] ==  np.unravel_index(i, (dim, dim))[0]:
                n_i.append(i-1)
                if not (i-1, i) in neighbours_list:
                    n_i_pair.append((i, i-1))
        except:
            pass
        try:
            if np.unravel_index(i+1, (dim, dim))[0] ==  np.unravel_index(i, (dim, dim))[0]:
                n_i.append(i+1)
                if not (i+1, i) in neighbours_list:
                    n_i_pair.append((i, i+1))
        except:
            pass
        neighbours[i] = n_i
        neighbours_list.extend(n_i_pair)
    neighbours_list = np.unique(np.array(neighbours_list), axis=0)
    return neighbours, neighbours_list

def brute_force_posterior(latent_dim, pgm, x, W, b):
    normaliser = []
    i = 0
    factors = pgm._factors
    num_samples = x.shape[0]
    for zs in product(*([[0, 1]] * latent_dim)): 
        log_singleton = singleton_fn(zs, factors).cpu() + ll_fn(zs, x, W) 
        log_pairwise = pairwise_fn(zs, factors).cpu() 
        log_joint_factor = joint_fn(zs, W, b) # 
        normaliser.append(log_singleton + log_pairwise + log_joint_factor)
        i += 1
    inds = torch.topk(torch.cat([n.reshape(-1, 1) for n in normaliser], dim=-1), k=1)[1].flatten()
    posterior_mode = torch.tensor(list(product(*([[0, 1]] * latent_dim))))[inds]
    return posterior_mode, normaliser, list(product(*([[0, 1]] * latent_dim)))

def singleton_fn(z, factors):
    latent_dim = len(z)
    singleton_potentials = torch.cat([f.get_potential().reshape(1, -1) for f in factors[:latent_dim]], dim=0)
    out = singleton_potentials[torch.arange(latent_dim), z].log().sum()
    return out

def pairwise_fn(z, factors):
    latent_dim = len(z)
    pairwise_potentials = torch.cat([f.get_potential()[None, ...] for f in factors[latent_dim:]], dim=0)
    out = 0.0
    out = out + pairwise_potentials[0][z[0], z[1]].log() # f(v_0, v_1)
    out = out + pairwise_potentials[1][z[1], z[2]].log() # f(v_1, v_2)
    out = out + pairwise_potentials[2][z[1], z[3]].log() # f(v_1, v_3)
    out = out + pairwise_potentials[3][z[2], z[4]].log() # f(v_2, v_4)
    out = out + pairwise_potentials[4][z[2], z[5]].log() # f(v_2, v_5)
    out = out + pairwise_potentials[5][z[3], z[6]].log() # f(v_3, v_6)
    out = out + pairwise_potentials[6][z[3], z[7]].log() # f(v_3, v_7)
    out = out + pairwise_potentials[7][z[4], z[8]].log() # f(z_4, z_8)
    out = out + pairwise_potentials[8][z[4], z[9]].log() # f(z_4, z_9)
    out = out + pairwise_potentials[9][z[5], z[10]].log() # f(z_5, z_10)
    out = out + pairwise_potentials[10][z[5], z[11]].log() # f(z_5, z_11)
    out = out + pairwise_potentials[11][z[6], z[12]].log() # f(z_6, z_12)
    out = out + pairwise_potentials[12][z[6], z[13]].log() # f(z_6, z_13)
    out = out + pairwise_potentials[13][z[7], z[14]].log() # f(z_7, z_14)
    out = out + pairwise_potentials[14][z[7], z[15]].log() # f(z_7, z_15)
    return out

def ll_fn(z, x, W):
    z = torch.FloatTensor(z)
    out = torch.einsum('l, nl->n', z, torch.einsum('nd, dl->nl', x.cpu(), W))
    return out

def joint_fn(z, W, b):
    z = torch.FloatTensor(z)
    terms = torch.exp(-torch.einsum('dl, l->d', W, z) + b)
    out = torch.sum(torch.log(terms) - torch.log(1+terms))
    return out

def compute_true_kl(prob_terms, z, logp, latent_pgm='tree', verbose=False):
    kl_div = 0.0
    if latent_pgm == 'tree':
        for i in range(len(z)):
            zi = torch.LongTensor(z[i])
            logprob_q = tree_density(prob_terms, zi, n=0).detach().cpu()
            prob_q = torch.exp(logprob_q)
            kl_div = kl_div + prob_q * (logprob_q - logp[i])
            if i % 2000 == 0:
                if verbose:
                    print(i, kl_div.mean())
    elif latent_pgm == 'MF':
        pass
    return kl_div

def compute_density(prob_terms, z, latent_pgm='tree'):
    sample_size, latent_dim = z.shape
    logprobs = 0.0
    if latent_pgm == 'tree':
        node_marginals, pairwise_marginals, _ = prob_terms
        logprobs = logprobs + (node_marginals['v_0'][torch.arange(sample_size), torch.LongTensor(z[:, 0])]+1e-31).log()
        pairwise_probs = ((pairwise_marginals['f(v_0, v_1)'] / (pairwise_marginals['f(v_0, v_1)'].sum(dim=-1, keepdim=True)+1e-31))\
            [torch.arange(sample_size), torch.LongTensor(z[:, 0]), torch.LongTensor(z[:, 1])]+1e-31).log()
        logprobs = logprobs + pairwise_probs
        next_ind_list = [torch.LongTensor(z[:, 1])]
        for i in range(int(np.log2(latent_dim-1))):
            next_ind_list_new = []
            for j in range(int(2**i)):
                pairwise_probs_1 = (pairwise_marginals[f'f(v_{int(2**i+j)}, v_{int((2**i+j)*2)})'] / \
                    (pairwise_marginals[f'f(v_{int(2**i+j)}, v_{int((2**i+j)*2)})'].sum(dim=-1, keepdim=True)+1e-31))\
                        [torch.arange(sample_size), next_ind_list[j], torch.LongTensor(z[:, int((2**i+j)*2)])]
                logprobs = logprobs + (pairwise_probs_1+1e-31).log()
                next_ind_list_new.append(torch.LongTensor(z[:, int((2**i+j)*2)]))
                pairwise_probs_2 = (pairwise_marginals[f'f(v_{int(2**i+j)}, v_{int((2**i+j)*2+1)})'] / \
                    (pairwise_marginals[f'f(v_{int(2**i+j)}, v_{int((2**i+j)*2+1)})'].sum(dim=-1, keepdim=True)+1e-31))\
                        [torch.arange(sample_size), next_ind_list[j], torch.LongTensor(z[:, int((2**i+j)*2+1)])]
                logprobs = logprobs + (pairwise_probs_2+1e-31).log()
                next_ind_list_new.append(torch.LongTensor(z[:, int((2**i+j)*2+1)]))
            next_ind_list = copy.deepcopy(next_ind_list_new)
    elif latent_pgm == 'MF':
        node_marginals = torch.cat([(1-prob_terms).reshape(-1, 1), prob_terms.reshape(-1, 1)], dim=-1)
        for i in range(latent_dim):
            logprobs = logprobs + node_marginals[torch.arange(sample_size), torch.LongTensor(z[:, i])].log()
    else:
        raise NotImplementedError
    return logprobs

def plot_smooth(y, temp=0.9):
    out = np.zeros_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = out[i-1] * temp + y[i] * (1-temp)
    return out

def sqrt_newton_schulz_autograd(A, numIters, dtype):
    batchSize = A.data.shape[0]
    dim = A.data.shape[1]
    normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A));
    I = Variable(torch.eye(dim,dim).view(1, dim, dim).
                 repeat(batchSize,1,1).type(dtype),requires_grad=False)
    Z = Variable(torch.eye(dim,dim).view(1, dim, dim).
                 repeat(batchSize,1,1).type(dtype),requires_grad=False)

    for i in range(numIters):
       T = 0.5*(3.0*I - Z.bmm(Y))
       Y = Y.bmm(T)
       Z = T.bmm(Z)
    sA = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    return sA

def simple_eig_sqrt(A):
    evals, evecs = torch.linalg.eig(A)
    return torch.matmul(evecs, torch.matmul(torch.diag(evals**(1/2)), torch.linalg.inv(evecs)))

class PotentialClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

def tree_density(prob_terms, z, n=0):
    node_marg, pair_marg, tree, = prob_terms
    z = {f'v_{i}': z[i] for i in range(len(z))}
    log_density = 0.0
    for i, f in enumerate(tree._factors):
        if len(f._rvs) == 1:
            log_density = log_density - (node_marg[f._rvs[0].__repr__()][n, z[f._rvs[0].__repr__()]]+1e-12).log() * (f._rvs[0].n_neighbours()-1)
        elif len(f._rvs) == 2:
            log_density = log_density + (pair_marg[f.__repr__()][n, z[f._rvs[0].__repr__()], z[f._rvs[1].__repr__()]]+1e-12).log()
        else:
            raise NotImplementedError
    return log_density

def conditional_from_marginal(mean, cov, z_prev):
    prec = torch.linalg.inv(cov)
    # std = cov[:, -1, -1].sqrt()
    std = 1/(prec[:, 0, 0].sqrt()+1e-10)
    mu = mean[:, 0, 0] - prec[:, 0, 1] * (z_prev - mean[:, 1, 0]) / (prec[:, 0, 0] + 1e-10)
    return mu, std

def weights_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.uniform_(-0.06, 0.06)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    
        
if __name__=='__main__':
    import pickle
    from factorgraph import Graph
    import time
    
    g = Graph(debug=False)
    for i in range(16):
        g.rv(f'v_{i}', 2)
    for i in range(16):
        if i == 0:
            g.factor([f'v_{i}'], potential=torch.FloatTensor([0.5, 0.5]))
        else:
            g.factor([f'v_{i}'], potential=torch.FloatTensor([1.0, 1.0]))
            
    g.factor(['v_0', 'v_1'], potential=torch.FloatTensor([
        [0.9, 0.1], 
        [0.8, 0.2]
    ]))
    g.factor(['v_1', 'v_2'], potential=torch.FloatTensor([
        [0.9, 0.1], 
        [0.7, 0.3]
    ]))
    g.factor(['v_1', 'v_3'], potential=torch.FloatTensor([
        [0.9, 0.1], 
        [0.7, 0.3]
    ]))
    for i in range(1, 3):
        for j in range(int(2**i)):
            g.factor([f'v_{int(2**i+j)}', f'v_{int((2**i+j)*2)}'], potential=torch.FloatTensor([
                [0.9, 0.1], 
                [0.7, 0.3]
            ]))
            g.factor([f'v_{int(2**i+j)}', f'v_{int((2**i+j)*2+1)}'], potential=torch.FloatTensor([
                [0.9, 0.1], 
                [0.7, 0.3]
            ]))
    iters, cvged = g.lbp(normalize=True)
    factors = g._factors
    
    b = 10
    N = 8
    W = np.zeros((N**2, 2*N))
    for i in range(N):
        for j in range(N):
            W[i*N+j, N+i] = 2 * b
            W[i*N+j, j] = 2 * b
    W = torch.FloatTensor(W)
    
    with open('datasets/bar_test_Tree_8_5000_10.0.pkl', 'rb') as file:
        x = pickle.load(file)
    file.close()
    x0 = torch.FloatTensor(x[:10]).reshape(10, -1)
    
    t0 = time.time()
    normaliser = brute_force_posterior(16, factors, x0, W, b)
    t1 = time.time()
    print(f'value: {normaliser} | time: {t1-t0:.2f}')
    
    