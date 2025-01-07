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


# Fix d = 2
# There are T periods
# 
# working sample row format:
# |  U_1  |  ...  |  U_T |  dif_01  |  ...  |  dif_(T-1)T  | C | w |
# 
#    (U_1, ..., U_T) has size 2T-1 or 2T (see below) and is the input to the potential functions (neural networks)
#       in the reduced dimension version:
#          U_1 is a number in the 1-d quantile domain [0,1]
#          U_2, ..., U_K are 2-dimensional vectors either in the quantile domain [0,1]^2 or in the problem domain R^2
#       in the full dimension version:
#          U_1, ..., U_K are 2-dimensional vectors either in the quantile domain [0,1]^2 or in the problem domain R^2
#    (dif_01, ..., dif_(T-1)T) has size 2T
#       dif_t(t+1) = X_t+1 - X_t in the problem domain R^2
#    C is the calculated cost of the sample
#    w is the weight
# 
# model contains the potential functions (NN's)
# Each potential function phi(1)_t or phi(2)_t takes a single dimensional input U(1)_t or U(2)_t.
# Each potential function h_t takes a multidimensional input U_1, ..., U_t
# Each potential functions yield a number calculated by the neural network
#
# In this example, we set T = 3


# extract data from working sample and apply model to find dual values
def mtg_parse(sample, model, d, T, monotone):
    
    phi_list, h_list = model
    size, n_cols = sample.shape
    
    u_size = T * d - monotone * (d - 1)
    dif_size = (T - 1) * d
    # check sample and model shapes
    assert len(phi_list) == u_size
    assert len(h_list) == dif_size
    assert n_cols == u_size + dif_size + 1 + 1
    
    # extract from the working sample
    u_list = [sample[:,i] for i in range(u_size)]
    dif_list = [sample[:,i+u_size] for i in range(dif_size)]
    c = sample[:, -2]
    w = sample[:, -1]
    
    # dual value
    D = sum([phi(u) for phi, u in zip(phi_list, u_list)])
    
    # martingale condition component
    # input to h must contain all previous u's
    H = []
    for t in range(1, T):
        Ut = sample[:, :(t * d - monotone * (d - 1))]   # cumulative u
        for i in range(d):
            h, dif = h_list.pop(0), dif_list.pop(0)
            H.append(h(Ut) * dif)
    H = sum(H)
    
    return D, H, c, w
    
# train loop
def mtg_train_loop(working_loader, model, d, T, monotone, beta, beta_multiplier, gamma, optimizer = None, verbose = 0):
    
    #   Primal:          max C
    #   Dual:            min D  st  D + H >= C
    #   Penalized dual:  min D + b(C - D - H)
    full_size = len(working_loader.dataset)
    
    # report
    if verbose > 0:
        print('   batch              D              H      deviation              P' + (not optimizer is None) * '                 loss')
        print('--------------------------------------------------------------------' + (not optimizer is None) * '---------------------')
    
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
        if not optimizer is None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
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
                      (not optimizer is None) * f'   {loss.item():18.4f}' +
                      f'    [{parsed:>7d}/{full_size:>7d}]')
        
    return _D.mean(), _H.mean(), _P.mean(), _D.std(), _H.std()

# main training function
def mtg_train(working_sample, model, d, T, monotone, opt_parameters, verbose = 0):

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
    
    lr =1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # iterative calls to train_loop
    D_series = []
    H_series = []
    P_series = []
    ds_series = []
    hs_series = []
    if verbose > 0:
        t0 = time.time() # timer
    for i in range(epochs):
        # if verbose > 0 and (i==0 or (i+1)%verbose == 0):
        print(f'epoch {i+1:4d}')
        verb = ((i+1)%verbose == 0 or (i+1 == epochs)) * 100
        if verb:
            print()
        D, H, P, ds, hs = mtg_train_loop(working_loader, model, d, T, monotone, beta, beta_multiplier, gamma, optimizer, verb)
        D_series.append(D)
        H_series.append(H)
        P_series.append(P)
        ds_series.append(ds)
        hs_series.append(hs)
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
        
    return model, D_series, H_series, P_series, ds_series, hs_series
    
def mtg_dual_value(working_sample, model, d, T, opt_parameters, normalize_pi = False):
    # global device
    if 'penalization' in opt_parameters.keys() and opt_parameters['penalization'] != 'L2':
        print('penalization not implemented: ' + opt_parameters['penalization'])
        return
    beta         = beta_L2
    beta_prime   = beta_L2_prime             # first derivative of L2 penalization function
    gamma        = opt_parameters['gamma']
    
    sample = torch.tensor(working_sample, device=device).float()
    phi, psi, h, L, C, w = mtg_parse(model, d, T, sample)
    D = phi.sum(axis=1) + psi.sum(axis=1)   # sum over dimensions
    H = (h * L).sum(axis=1)                 # sum over dimensions
    deviation = C - D - H
    P = beta(deviation, gamma)
    pi_star = w * beta_prime(deviation, gamma)
    sum_pi_star = pi_star.sum()
    if normalize_pi and sum_pi_star > 0:
        pi_star = pi_star / sum_pi_star
    
    return D.detach().mean().cpu().numpy(), P.detach().mean().cpu().numpy(), pi_star.detach().cpu().numpy()


def generate_model(d, T, monotone):
    hidden_size = 64
    n_hidden_layers = 2
    if monotone:
        phi_list = nn.ModuleList([PotentialF(1, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(1 + d * (T-1))])
    else:
        phi_list = nn.ModuleList([PotentialF(1, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d * T)])
    h_list   = nn.ModuleList([PotentialF(1, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d * (T-1))])
    model = nn.ModuleList([phi_list, h_list])
    return model


# from a sample of (u,v) in the domain [0,1]^d x [0,1]^d
# u maps to x and v maps to y through the inverse cumulatives
def generate_working_sample(u, v, x, y, c, weight = None):
    size = len(u)
    dif = y - x          # each column i has (yi - xi)
    if weight is None:
        weight = np.ones(size) / size
    working_sample = np.hstack([u, v, dif, c.reshape(size, 1), weight.reshape(size, 1)])
    return working_sample



# -----------------------------------------------

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


