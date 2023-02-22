# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:35:25 2021
@author: souzam
PyTorch implementation of Eckstein and Kupper 2019 - Computation of Optimal Transport...
"""

import numpy as np
import itertools
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import norm

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'   # pytorch version issues

# class for auxiliary dataset
class QDataset(Dataset):
    def __init__(self, d, n):
        self.X = X
        self.Y = Y
       
    def __len__(self):
        return len(self.X)
       
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
 
def generate_tensors(sample_mu_X, sample_mu_Y, sample_th_X, sample_th_Y):
    t_mu_X = torch.tensor(sample_mu_X).float()
    t_mu_Y = torch.tensor(sample_mu_Y).float()
    t_th_X = torch.tensor(sample_th_X).float()
    t_th_Y = torch.tensor(sample_th_Y).float()
    return t_mu_X, t_mu_Y, t_th_X, t_th_Y

def generate_loaders(sample_mu_X, sample_mu_Y, sample_th_X, sample_th_Y, batch_size, shuffle = True):
    t_mu_X, t_mu_Y, t_th_X, t_th_Y = generate_tensors(sample_mu_X, sample_mu_Y, sample_th_X, sample_th_Y)
    mu_dataset = SampleDataset(t_mu_X, t_mu_Y)
    th_dataset = SampleDataset(t_th_X, t_th_Y)
    mu_loader = DataLoader(mu_dataset, batch_size = batch_size, shuffle = shuffle)
    th_loader = DataLoader(th_dataset, batch_size = batch_size, shuffle = shuffle)
    return mu_loader, th_loader
    
# class for model for each hj (or gj) to be minimized (rhs of eq 2.8)
class Phi(nn.Module):
 
    def __init__(self, input_dimension, n_hidden_layers = 2, hidden_size = 32):
        super(Phi, self).__init__()
        layers = [nn.Linear(input_dimension, hidden_size), nn.ReLU()]
        for i in range(n_hidden_layers):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers += [nn.Linear(hidden_size, 1)]
        self.hi = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.hi(x)
n = 4
d = 2
print(f'grid size {n*d:d}')

# quantile grids for x and y
batch_size = int(np.sqrt(n*d))   # no special formula for this...
q_set = np.array(list(itertools.product(*[list(range(n)) for i in range(d)])))
qx_set = torch.tensor(q_set)
qy_set = torch.tensor(q_set)
qx_loader = DataLoader(x_set, batch_size = batch_size, shuffle = True)
qy_loader = DataLoader(qy_set, batch_size = batch_size, shuffle = True)
 

qx = q_set[5]
qy = q_set[10]

# domain-specific values: cost function, Lk functions (k = 1, ..., d) and theta probability
# arguments are quantiles, must be converted to the interval [0,1] and then to the problem
#    domain the using inverse cummulative function

x_normal_scale = [1.0, 1.0]
y_normal_scale = [1.5, 2.0]

def inv_x(xhat, i):
    return norm.ppf(xhat) * x_normal_scale[i]

def inv_y(xhat, i):
    return norm.ppf(xhat) * y_normal_scale[i]

def cost_f(qx, qy):
    xhat = (2 * qx + 1) / (2 * n)   # a point in the 1-hypercube
    yhat = (2 * qy + 1) / (2 * n)   # a point in the 1-hypercube
    x = np.array([inv_x(xhat[:,i], i) for i in range(d)]).T
    y = np.array([inv_y(yhat[:,i], i) for i in range(d)]).T
    return (x * y).sum(axis=1)
     

# penalty function 
def beta_Lp(x, p, gamma):
    return (1 / gamma) * (1 / p) * torch.pow(torch.relu(gamma * x), p)
 
def beta_L2(x, gamma):
    return beta_Lp(x, 2, gamma)
    # beta(x) = gamma * 1/2 * x^2
 
def beta_L2_prime(x, gamma):
    return gamma * torch.relu(x)
    # beta'(x) = gamma * (x)+
    

# main function
def train_loop(cost, mu_loader, th_loader,
               phi_x_list, phi_y_list, h_list, beta, gamma,
               optimizer = None, verbose = False):
    
    # f:        cost function to be maximized (primal)
    # phi_list: list of potential functions as neural networks (dual)
    # beta:     penalization function
    
    full_size = len(mu_loader.dataset)
    if verbose:
        if optimizer is None:
            print('       sum_mu[phi]           sum_mu[h]     sum_th[penalty]           deviation')
        else:
            print('       sum_mu[phi]           sum_mu[h]     sum_th[penalty]                loss')
    _value     = []
    _penalty   = []
    # for batch, ((_mu_X, _mu_Y), (_th_X, _th_Y)) in enumerate(zip(mu_loader, th_loader)): break   # test mode only
    for batch, ((_mu_X, _mu_Y), (_th_X, _th_Y)) in enumerate(zip(mu_loader, th_loader)):
        size, d = _mu_X.shape
        
        # value integral, on mu
        phi_x_values_mu = [phi(_mu_X[:,i].view(size, 1))[:,0] for i, phi in enumerate(phi_x_list)]
        phi_y_values_mu = [phi(_mu_Y[:,i].view(size, 1))[:,0] for i, phi in enumerate(phi_y_list)]
        phi_x_mu  = sum(phi_x_values_mu)         # array of values
        phi_y_mu  = sum(phi_y_values_mu)         # array of values
        value = torch.mean(phi_x_mu) + torch.mean(phi_y_mu)  # "integral"
        
        # for show only (we will only look at h in the penalization integral below)
        h_values_mu = [phi(_mu_X.view(size, d))[:,0] * (_mu_Y[:,i] - _mu_X[:,i]) for i, phi in enumerate(h_list)]
        h_mu    = sum(h_values_mu)   # array of values
        value_h = torch.mean(h_mu)   # "integral"
        
        # penalization integral, on th
        cost_th = cost(_th_X, _th_Y)
        phi_x_values_th = [phi(_th_X[:,i].view(size, 1))[:,0] for i, phi in enumerate(phi_x_list)]
        phi_y_values_th = [phi(_th_Y[:,i].view(size, 1))[:,0] for i, phi in enumerate(phi_y_list)]
        h_values_th = [phi(_th_X.view(size, d))[:,0] * (_th_Y[:,i] - _th_X[:,i]) for i, phi in enumerate(h_list)]
        phi_x_th = sum(phi_x_values_th)         # array of values
        phi_y_th = sum(phi_y_values_th)         # array of values
        h_th     = sum(h_values_th)             # array of values
        dual_th = phi_x_th + phi_y_th + h_th    # array of values
        deviation = (cost_th - dual_th)  # array of values
        penalty = torch.mean(beta(deviation, gamma))   # "integral"
        
        _value.append(value.item())
        # _value.append(value.item() + value_h.item())   # if intended to include h_mu
        _penalty.append(penalty.item())
        
        # backpropagation
        if not optimizer is None:
            loss = value + penalty
            # loss = (value + value_h) + penalty   # if intended to include h_mu
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # iteration report
        if verbose:
            c = (batch + 1) * size
            if batch == 0 or c % 100000 == 0:
                if optimizer is None:
                    print(f'{value.item():18.4f}' + \
                        f'  {value_h.item():18.4f}' + \
                        f'  {penalty.item():18.4f}' + \
                        f'  {torch.mean(deviation).item():18.4f}')
                else:
                    print(f'{value.item():18.4f}' + \
                        f'  {value_h.item():18.4f}' + \
                        f'  {penalty.item():18.4f}' + \
                        f'  {loss.item():18.4f}    [{c:>7d}/{full_size:>7d}]')
                        
                    
    return np.mean(_value), np.std(_value), np.mean(_penalty)
