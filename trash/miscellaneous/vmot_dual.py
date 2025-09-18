# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:35:25 2021
@author: souzam
Dual approximation of VMOT using neural networks
References
- Hiew, Lim Pass, Souza "Modularity, Convex conjugates and VMOT" (in development)
- Eckstein and Kupper "Computation of Optimal Transport" (2021)
"""

import numpy as np
import matplotlib.pyplot as pl
import datetime as dt, time
import pickle
import itertools

import torch
from torch import nn
from torch.utils.data import DataLoader

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'   # pytorch version issues


# CUDA if available
use_cuda = True   # if available...
device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
print('Using device:', device)

'''
1. Use the ultils functions to construct a "working sample" that includes
    - x and y (or u, v) which are used as inputs to the neural networks
    - dif where dif_ = y_i - x_i
    - c = cost
    - w = weight (associated with the sample element, typically 1/n)
    
2. Call mtg_optimize, which in turn
    - receives a preexisting or creates a new "model" (set of potential functions)
    - calls the train_loop to optimize the neural networks
'''

# penalty function and its derivative
def beta_Lp(x, p, gamma):
    return (1 / gamma) * (1 / p) * torch.pow(torch.relu(gamma * x), p)
 
def beta_L2(x, gamma):
    return beta_Lp(x, 2, gamma)
 
def beta_L2_prime(x, gamma):
    return gamma * torch.relu(x)
    

# base class for each potential function phi, psi or h
class PotentialF(nn.Module):
    def __init__(self, input_dimension, n_hidden_layers = 2, hidden_size = 64):
        super(PotentialF, self).__init__()
        layers = [nn.Linear(input_dimension, hidden_size), nn.ReLU()]
        for i in range(n_hidden_layers):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers += [nn.Linear(hidden_size, 1)]
        self.hi = nn.Sequential(*layers)
    def forward(self, x):
        return self.hi(x)


# not used for this working sample format
# def _participating_cols(t, d, T, monotone):
#     if monotone:
#         ans = [0]
#         for j in range(d):          # ex. for d=2, j=0 means u and j=1 means v
#             for k in range(1, t+1):
#                 ans.append(1 + j * (T-1) + (k-1))
#     else:
#         ans = []
#         for j in range(d):          # ex. for d=2, j=0 means u and j=1 means v
#             for k in range(t+1):
#                 ans.append(j * T + k)
                
#     return ans

# _participating_cols(t=0, d=2, T=3, monotone=True)
# _participating_cols(t=1, d=2, T=3, monotone=True)
# _participating_cols(t=2, d=2, T=3, monotone=True)

# _participating_cols(t=0, d=2, T=3, monotone=False)
# _participating_cols(t=1, d=2, T=3, monotone=False)
# _participating_cols(t=2, d=2, T=3, monotone=False)

# Fix d = 2
# There are T periods
# In this example, we set d = 2, T = 3
# working sample row format:
# monotone:        u0,     u1, v1, u2, v2, dif_x12, dif_y12, dif_x23, dif_y23, c, w
# full dimension:  u0, v0, u1, v1, u2, v2, dif_x12, dif_y12, dif_x23, dif_y23, c, w

# extract data from working sample and apply model to find dual values
def mtg_parse(sample, model, d, T, monotone):
    
    phi_list, h_list = model
    size, n_cols = sample.shape
    
    u_size = T * d - monotone * (d - 1)
    dif_size = (T - 1) * d
    # check sample and model shapes
    assert len(phi_list) == u_size
    assert len(h_list) == dif_size
    assert n_cols == u_size + dif_size + 1 + 1, f'n_cols {n_cols}, u_size {u_size}, dif_size {dif_size}'
    
    # extract from the working sample
    u_list = [sample[:,i] for i in range(u_size)]
    dif_list = [sample[:,i+u_size] for i in range(dif_size)]
    c = sample[:, -2]
    w = sample[:, -1]
    
    # dual value
    D = sum([phi(u[:,None]) for phi, u in zip(phi_list, u_list)])
    D = D.reshape(size)
    # martingale condition component
    # input to h must contain all previous U's
    H = []
    count = 0
    for t in range(1, T):
        if monotone:
            Ut = sample[:, :1 + (t-1) * d]
        else:
            Ut = sample[:, :t * d]
        for i in range(d):
            h, dif = h_list[count], dif_list[count]
            H.append(h(Ut).reshape(size) * dif)
            count += 1
    assert count == len(h_list)
    H = sum(H)
    
    return D, H, c, w
    
# train loop
def mtg_train_loop(working_loader, model, opt, d, T, monotone, beta, beta_multiplier, gamma, verbose = 0):
    
    #   Primal:          max C
    #   Dual:            min D  st  D + H >= C
    #   Penalized dual:  min D + b(C - D - H)
    full_size = len(working_loader.dataset)
    
    # report
    if verbose > 0:
        print('   batch              D              H      deviation              P' + (not opt is None) * '                 loss')
        print('--------------------------------------------------------------------' + (not opt is None) * '---------------------')
    
    _D = np.array([])   # dual value
    _H = np.array([])   # mtg component - should converge to zero when (mu) <= (nu)
    _P = np.array([])   # penalty
    
    # for batch, sample in enumerate(sample_loader): break
    for batch, sample in enumerate(working_loader):
        
        # time series of dual value, mtg component and penalization
        D, H, c, w = mtg_parse(sample, model, d, T, monotone)
        deviation = c - D - H
        penalty = beta(deviation, gamma)
        
        # loss and backpropagation
        loss = (1 / full_size) * D.sum() + beta_multiplier * (w * penalty).sum()
        if not opt is None:
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        # record evolution
        _D = np.append(_D, D.detach().cpu().numpy())
        _H = np.append(_H, H.detach().cpu().numpy())
        _P = np.append(_P, penalty.detach().cpu().numpy())
        
        # report
        parsed = len(_D)
        if verbose > 0 and (parsed == full_size or (batch+1) % verbose == 0):
                print(f'{batch+1:8d}' + 
                      f'   {D.mean().item():12.4f}' +
                      f'   {H.mean().item():12.4f}' +
                      f'   {deviation.mean().item():12.4f}' +
                      f'   {penalty.mean().item():12.4f}' +
                      (not opt is None) * f'   {loss.item():18.4f}' +
                      f'    [{parsed:>7d}/{full_size:>7d}]')
        
    return _D.mean(), _H.mean(), _P.mean(), _D.std(), _H.std()

# main training function
def mtg_train(working_sample, model, opt, d, T, monotone, opt_parameters, verbose = 0):

    # check inputs
    if 'penalization' in opt_parameters.keys() and opt_parameters['penalization'] != 'L2':
        print('penalization not implemented: ' + opt_parameters['penalization'])
        return
    beta            = beta_L2                             # L2 penalization is the only one available
    beta_multiplier = opt_parameters['beta_multiplier']
    gamma           = opt_parameters['gamma']
    batch_size      = opt_parameters['batch_size']
    epochs          = opt_parameters['epochs']
    
    # loader
    shuffle        = True     # must be True to avoid some bias towards the last section of the quantile grid
    working_sample = torch.tensor(working_sample, device=device).float()
    working_loader = DataLoader(working_sample, batch_size = batch_size, shuffle = shuffle)
    
    # iterative calls to train_loop
    D_series = []
    if verbose > 0:
        t0 = time.time() # timer
    for i in range(epochs):
        # if verbose > 0 and (i==0 or (i+1)%verbose == 0):
        print(f'epoch {i+1:4d}')
        verb = ((i+1)%verbose == 0 or (i+1 == epochs)) * 100
        if verb:
            print()
        D, H, P, ds, hs = mtg_train_loop(working_loader, model, opt, d, T, monotone, beta, beta_multiplier, gamma, verb)
        D_series.append(D)
        if verb:
            print('\nmeans')
            print(f'   D   = {D:12.4f}')
            print(f'   H   = {H:12.4f}')
            print(f'   P   = {P:12.4f}\n')
            print(f'   D std = {ds:12.4f}')
            print(f'   H std = {hs:12.4f}')
            print()
    if verbose > 0:
        t1 = time.time() # timer
        print('duration = ' + str(dt.timedelta(seconds=round(t1 - t0))))
        print()
        
    return D_series
    
def mtg_dual_value(working_sample, model, d, T, monotone, opt_parameters, normalize_pi = False):
    # global device
    if 'penalization' in opt_parameters.keys() and opt_parameters['penalization'] != 'L2':
        print('penalization not implemented: ' + opt_parameters['penalization'])
        return
    beta         = beta_L2
    beta_prime   = beta_L2_prime             # first derivative of L2 penalization function
    gamma        = opt_parameters['gamma']
    
    sample = torch.tensor(working_sample, device=device).float()
    D, H, c, w = mtg_parse(sample, model, d, T, monotone)
    deviation = c - D - H
    penalty = beta(deviation, gamma)
    pi_star = w * beta_prime(deviation, gamma)
    sum_pi_star = pi_star.sum()
    if normalize_pi and sum_pi_star > 0:
        pi_star = pi_star / sum_pi_star
    
    return D.detach().mean().cpu().numpy(), penalty.detach().mean().cpu().numpy(), pi_star.detach().cpu().numpy()

def generate_model(d, T, monotone):
    hidden_size = 64
    n_hidden_layers = 2
    if monotone:
        phi_list = nn.ModuleList([PotentialF(1, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size).to(device) for i in range(1 + d * (T-1))])
        h_list   = nn.ModuleList([PotentialF(1 + (j-1) * d, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size).to(device) for j in range(1, T) for k in range(d)])
    else:
        phi_list = nn.ModuleList([PotentialF(1, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size).to(device) for i in range(d * T)])
        h_list   = nn.ModuleList([PotentialF(j * d, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size).to(device) for j in range(1, T) for k in range(d)])
    model = nn.ModuleList([phi_list, h_list])
    lr =1e-4
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    return model, opt

# [1 + (j-1) * d for j in range(1, T) for k in range(d)]
# [j * d for j in range(1, T) for k in range(d)]
# from a sample of (u,v) in the domain [0,1]^d x [0,1]^d
# u maps to x and v maps to y through the inverse cumulatives
# In this example, we set d = 2, T = 3
# full dimension:    u0, v0, u1, v1, u2, v2, dif_x12, dif_y12, dif_x23, dif_y23, c, w
# reduced dimension: u0,     u1, v1, u2, v2, dif_x12, dif_y12, dif_x23, dif_y23, c, w
def generate_working_sample(u, v, x, y, c, weight = None):
    size = len(u)
    if weight is None:
        weight = np.ones(size) / size
    nperiods = x.shape[1]
    x_dif = [x[:,i+1] - x[:,i] for i in range(nperiods - 1)]
    y_dif = [y[:,i+1] - y[:,i] for i in range(nperiods - 1)]
    dif = np.vstack(x_dif + y_dif).T
    working_sample = np.hstack([u, v, dif, c[:,None], weight.reshape(size, 1)])
    return working_sample



# -----------------------------------------------
# to do: use pytorch serialization scheme instead of pickle

# utils - file dump
_dir = './model_dump/'
_file_prefix = 'results_'
_file_suffix = '.pickle'

def dump_results(results, label='test'):
    # move to cpu before dumping
    cpu_device = torch.device('cpu')
    for i in range(len(results)):
        if isinstance(results[i], torch.nn.modules.container.ModuleList):
            # print(i, type(results[i]))
            results[i] = results[i].to(cpu_device)
    
    # dump
    _path = _dir + _file_prefix + label + _file_suffix
    with open(_path, 'wb') as file:
        pickle.dump(results, file)
    print('model saved to ' + _path)

def load_results(label=''):
    _path = _dir + _file_prefix + label + _file_suffix
    with open(_path, 'rb') as file:
        results = pickle.load(file)
    print('model loaded from ' + _path)
    for i in range(len(results)):
        if isinstance(results[i], torch.nn.modules.container.ModuleList):
            # print(i, type(results[i]))
            results[i] = results[i].to(device)
    return results

# utils - convergence plots
def convergence_plot(value_series_list, labels, h_series_list=None,
                     ref_value=None, title='Numerical value - convergence'):
    ref_color='black'
    ref_label='reference'
    pl.figure(figsize = [7,7])   # plot in two iterations to have a clean legend
    for v in value_series_list:
        pl.plot(range(1, len(v)+1), v)
    if not ref_value is None:
        pl.axhline(ref_value, linestyle=':', color=ref_color)
    pl.legend(labels + [ref_label])
    if not h_series_list is None:
        pl.gca().set_prop_cycle(None)   # reset color cycler
        for v, h in zip(value_series_list, h_series_list):
            pl.plot(range(1, len(v)+1), v+h, linestyle=':')
    # pl.xlabel('epoch')
    pl.title(title)
    pl.annotate('lower bound', (len(v)*3/4, ref_value-500), color='darkgrey')   # trial and error to find a good position
    pl.show()

# utils - convergence plots
def convergence_plot_empirical(value_series_list, labels, h_series_list=None,
                               lower_bound=None, title='Numerical value - convergence'):
    pl.figure(figsize = [5,5])   # plot in two iterations to have a clean legend
    for v in value_series_list:
        pl.plot(range(1, len(v)+1), v)
    if not lower_bound is None:
        pl.fill_between(pl.gca().get_xlim(), y1=lower_bound, y2=0, alpha = .5, facecolor = 'lightgrey')
        pl.annotate('lower bound', (len(v)*2/3, lower_bound-400), color='darkgrey')   # trial and error to find a good position
        # pl.axhline(lower_bound, linestyle=':', color='lightgrey')
    pl.legend(labels)
    if not h_series_list is None:
        pl.gca().set_prop_cycle(None)   # reset color cycler
        for v, h in zip(value_series_list, h_series_list):
            pl.plot(range(1, len(v)+1), v+h, linestyle=':')
    pl.title(title)
    pl.show()


