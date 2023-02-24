# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:35:25 2021
@author: souzam
PyTorch implementation of Eckstein and Kupper 2019 - Computation of Optimal Transport...
"""

import numpy as np
import matplotlib.pyplot as pl
import itertools
import torch
from torch import nn
from torch.utils.data import DataLoader

import datetime as dt, time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'   # pytorch version issues



# penalty function 
def beta_Lp(x, p, gamma):
    return (1 / gamma) * (1 / p) * torch.pow(torch.relu(gamma * x), p)
 
def beta_L2(x, gamma):
    return beta_Lp(x, 2, gamma)
 
def beta_L2_prime(x, gamma):
    return gamma * torch.relu(x)
    
# class for model for each hj (or gj) to be minimized (rhs of eq 2.8)
class PotentialF(nn.Module):
 
    def __init__(self, input_dimension, n_hidden_layers = 2, hidden_size = 32):
        super(PotentialF, self).__init__()
        layers = [nn.Linear(input_dimension, hidden_size), nn.ReLU()]
        for i in range(n_hidden_layers):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers += [nn.Linear(hidden_size, 1)]
        self.hi = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.hi(x)
    
def mtg_parse(sample, phi_list, psi_list, h_list):
    # expected format:
    # |-- X -- | -- Y -- | -- L -- | c | th |
    size, num_cols = sample.shape
    d = int((num_cols - 2) / 3)
    
    phi = torch.hstack([phi(sample[:,i].view(size, 1)) for i, phi in enumerate(phi_list)])
    psi = torch.hstack([psi(sample[:,i+d].view(size, 1)) for i, psi in enumerate(psi_list)])
    h   = torch.hstack([h(sample[:,:d]) for i, h in enumerate(h_list)])
    L     = sample[:,2*d:3*d]
    c     = sample[:,3*d]
    theta = sample[:,3*d+1]
    
    return phi, psi, h, L, c, theta
    
# train loop
def mtg_train_loop(working_loader, phi_list, psi_list, h_list,
                 b_multiplier, beta, gamma, optimizer = None, verbose = 0):
    full_size = len(working_loader.dataset)
    if verbose > 0:
        print('   batch              D              H      deviation              P' + (not optimizer is None) * '                 loss')
        print('--------------------------------------------------------------------' + (not optimizer is None) * '---------------------')
    
    _D = np.array([])   # dual value
    _H = np.array([])   # should converge to zero if distributions are in convex order (for report purposes only)
    _P = np.array([])   # penalty
    # for batch, subsample in enumerate(sample_loader): break
    for batch, sample in enumerate(working_loader):
        
        # dual value and penalization
        phi, psi, h, L, c, theta = mtg_parse(sample, phi_list, psi_list, h_list)
        D = (phi + psi).sum(axis=1)   # sum over dimensions
        H = (h * L).sum(axis=1)       # sum over dimensions
        deviation = D + H - c
        P = (theta / theta.sum()) * beta(deviation, gamma)
        
        _D = np.append(_D, D.detach().numpy())
        _H = np.append(_H, H.detach().numpy())
        _P = np.append(_P, P.detach().numpy())
        
        # loss and backpropagation
        loss = (-D + b_multiplier * P).mean()
        if not optimizer is None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # iteration report
        parsed = len(_D)
        if verbose > 0 and (parsed == full_size or (batch+1) % verbose == 0):
                print(f'{batch+1:8d}' + 
                      f'   {D.mean().item():12.4f}' +
                      f'   {H.mean().item():12.4f}' +
                      f'   {deviation.mean().item():12.4f}' +
                      f'   {P.mean().item():12.4f}' +
                      (not optimizer is None) * f'   {loss.item():18.4f}' +
                      f'    [{parsed:>7d}/{full_size:>7d}]')
        
    return _D.mean(), _D.std(), _H.mean(), _P.mean()

def mtg_train(working_sample, opt_parameters, model = None, verbose = False):
    
    # check inputs
    n, num_cols = working_sample.shape
    d = int((num_cols - 2) / 3)
    if 'penalization' in opt_parameters.keys() and opt_parameters['penalization'] != 'L2':
        print('penalization not implemented: ' + opt_parameters['penalization'])
        return
    beta         = beta_L2                   # L2 penalization
    b_multiplier = opt_parameters['b_multiplier']
    gamma        = opt_parameters['gamma']
    batch_size   = opt_parameters['batch_size']
    macro_epochs = opt_parameters['macro_epochs']
    micro_epochs = opt_parameters['micro_epochs']
    
    # loader
    shuffle    = True     # must be True to avoid some bias towards the last section of the quantile grid
    working_sample = torch.tensor(working_sample).float()
    working_loader = DataLoader(working_sample, batch_size = batch_size, shuffle = shuffle)
    
    # modules and optimizers
    lr =1e-4
    hidden_size = 32
    n_hidden_layers = 2
    if model is None:
        phi_list = nn.ModuleList([PotentialF(1, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d)])
        psi_list = nn.ModuleList([PotentialF(1, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d)])
        h_list   = nn.ModuleList([PotentialF(d, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d)])
        model = [phi_list, psi_list, h_list]
    else:
        phi_list, psi_list, h_list = model
    optimizer = torch.optim.Adam(list(phi_list.parameters()) + list(psi_list.parameters()) + list(h_list.parameters()), lr=lr)
    
    # iterative calls to train_loop
    D_series = []
    s_series = []
    H_series = []
    P_series = []
    if verbose > 0:
        t0 = time.time() # timer
    for i in range(macro_epochs):
        for j in range(micro_epochs):
            verb = (j + 1 == micro_epochs) * verbose
            if verbose > 0 and j+1 % 10 == 0:
                print(f'{i+1:4d}, {j+1:3d}')
            if verb:
                print()
            D, s, H, P = mtg_train_loop(working_loader, phi_list, psi_list, h_list, b_multiplier, beta, gamma, optimizer, verb)
            D_series.append(D)
            s_series.append(D)
            H_series.append(H)
            P_series.append(P)
            if verb:
                print('\nmeans')
                print(f'   D   = {D:12.4f}')
                print(f'   std = {s:12.4f}')
                print(f'   H   = {H:12.4f}')
                print(f'   P   = {P:12.4f}\n')
    if verbose > 0:
        t1 = time.time() # timer
        print('duration = ' + str(dt.timedelta(seconds=round(t1 - t0))))
        
    return model, D_series, s_series, H_series, P_series
    
def dual_value(working_sample, opt_parameters, model):
    if 'penalization' in opt_parameters.keys() and opt_parameters['penalization'] != 'L2':
        print('penalization not implemented: ' + opt_parameters['penalization'])
        return
    beta_prime   = beta_L2_prime   # first derivative of L2 penalization function
    gamma        = opt_parameters['gamma']
    phi_list, psi_list, h_list = model
    
    working_sample = torch.tensor(working_sample).float()
    phi, psi, h, L, c, theta = mtg_parse(working_sample, phi_list, psi_list, h_list)
    D = (phi + psi).sum(axis=1)   # sum over dimensions
    H = (h * L).sum(axis=1)       # sum over dimensions
    deviation = D + H - c
    pi_star = (theta * beta_prime(deviation, gamma)).detach().numpy()
    
    return D.mean().item(), H.mean().item(), pi_star

# working sample format:
# -- X -- | -- Y -- | -- L -- | c | th |

def generate_working_sample(xy_sample, cost_f, uniform_theta = True, theta_prob = None):
    size, num_cols = xy_sample.shape
    d = int(num_cols / 2)
    x = xy_sample[:, :d]
    y = xy_sample[:, d:]
    l_set = y - x           # a matrix with d columns, each column has (yi - xi)
    c_set = cost_f(x, y)   # a vector of costs
    if uniform_theta:
        theta = np.ones(size) / size
    else:
        if theta is None:
            print('a theta probability must be specified')
            return
        theta = theta_prob(x, y)   # joint probability of each quantile
    
    # check and build working sample
    working_sample = np.hstack([xy_sample, l_set, c_set.reshape(size, 1), theta.reshape(size, 1)])
    assert working_sample.shape[1] == 3 * d + 2
    return working_sample
    
def generate_working_sample_marginals(xi_sample_list, yi_sample_list, cost_f,
                                      monotone_x = False,
                                      uniform_theta = True,
                                      theta_prob = None):
    if monotone_x:
        # order xi's, shuffle together and separate again
        xi_sample_list = [np.sort(xi) for xi in xi_sample_list]
        x_sample = np.vstack(xi_sample_list).T
        np.random.shuffle(x_sample)
        # xi_sample_list = [x_sample[:,i] for i in range(x_sample.shape[1])]
        # couple each instance of x as a block with all possible combinations of yi's
        # y_combination = np.array(list(itertools.product(*yi_sample_list)))
        marginals_list = list(x_sample) + yi_sample_list
        xy_sample = np.array(list(itertools.product(*marginals_list)))
    else:
        # all combinations of quantiles on x cross y
        marginals_list = xi_sample_list + yi_sample_list
        xy_sample = np.array(list(itertools.product(*marginals_list)))
    return generate_working_sample(xy_sample, cost_f, uniform_theta, theta_prob)
        
    
def generate_working_sample_quantile(n, d, inv_cum_x, inv_cum_y, cost_f,
                                     monotone_x = False,
                                     uniform_theta = True,
                                     theta_prob = None):

    # all combinations of quantiles on x cross y
    if monotone_x:
        q_set = np.array(list(itertools.product(*[list(range(n)) for i in range(d+1)])))
        q_set = np.hstack([np.tile(q_set[:,0], (d-1, 1)).T, q_set])
    else:
        q_set = np.array(list(itertools.product(*[list(range(n)) for i in range(2 * d)])))
    qx = q_set[:, :d]
    qy = q_set[:, d:]
    xhat = (2 * qx + 1) / (2 * n)   # points in the d-hypercube
    yhat = (2 * qy + 1) / (2 * n)   # points in the d-hypercube
    x = np.array([inv_cum_x(xhat[:,i], i) for i in range(d)]).T
    y = np.array([inv_cum_y(yhat[:,i], i) for i in range(d)]).T
    l_set = y - x           # a matrix with d columns, each column has (yi - xi)
    c_set = cost_f(x, y)   # a vector of costs
    if uniform_theta:
        theta = np.ones(len(q_set)) / len(q_set)
    else:
        if theta is None:
            print('a theta probability must be specified')
            return
        theta = theta_prob(q_set, coupling = 'independent')   # joint probability of each quantile
    
    # check and build working sample
    working_sample = np.hstack([q_set, l_set, c_set.reshape(len(theta), 1), theta.reshape(len(theta), 1)])
    return working_sample

def plot_sample_2d(sample, label='sample'):
    figsize = [12,12]
    X1, X2, Y1, Y2 = sample[:,0], sample[:,1], sample[:,2], sample[:,3]
    pl.figure(figsize=figsize)
    pl.title(label)
    pl.axis('equal')
    pl.xlabel('X Y 1')
    pl.ylabel('X,Y 2')
    pl.scatter(Y1, Y2, alpha=.05)
    pl.scatter(X1, X2, alpha=.05)
    pl.legend(['X sample', 'Y sample'])
    