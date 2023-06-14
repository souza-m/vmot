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
# import pandas as pd
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
    - L where l_i = y_i - x_i
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
    def __init__(self, input_dimension, n_hidden_layers = 2, hidden_size = 32):
        super(PotentialF, self).__init__()
        layers = [nn.Linear(input_dimension, hidden_size), nn.ReLU()]
        for i in range(n_hidden_layers):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers += [nn.Linear(hidden_size, 1)]
        self.hi = nn.Sequential(*layers)
    def forward(self, x):
        return self.hi(x)


# working sample format:
# | -- X -- | -- Y -- | -- L -- | C | w |

# extract from working sample
def mtg_parse(model, sample):
    
    phi_list, psi_list, h_list = model
    size, num_cols = sample.shape
    nx, ny = len(phi_list), len(psi_list)    # input lengths (depend on dimensionality)
    d = num_cols - nx - ny - 2
    
    # extract from the working sample
    u = sample[:, : nx]
    v = sample[:, nx : nx + ny]
    L = sample[:, nx + ny : nx + ny + d]
    C = sample[:, nx + ny + d]
    w = sample[:, nx + ny + d + 1]
    
    # calculate using model
    phi = torch.hstack([phi(u[:, i].view(size, 1)) for i, phi in enumerate(phi_list)])
    psi = torch.hstack([psi(v[:, j].view(size, 1)) for j, psi in enumerate(psi_list)])
    h   = torch.hstack([h(u) for h in h_list])
    
    return phi, psi, h, L, C, w
    
# train loop
def mtg_train_loop(model, working_loader, beta, beta_multiplier, gamma, optimizer = None, verbose = 0):
    
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
        phi, psi, h, L, C, w = mtg_parse(model, sample)
        D = phi.sum(axis=1) + psi.sum(axis=1)   # sum over dimensions
        H = (h * L).sum(axis=1)       # sum over dimensions
        deviation = C - D - H
        P = beta(deviation, gamma)
        _D = np.append(_D, D.detach().cpu().numpy())
        _H = np.append(_H, H.detach().cpu().numpy())
        _P = np.append(_P, P.detach().cpu().numpy())
        
        # loss and backpropagation
        # loss = (-D + b_multiplier * P).mean()
        loss = (1 / full_size) * D.sum() + beta_multiplier * (w * P).sum()
        if not optimizer is None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # report
        parsed = len(_D)
        if verbose > 0 and (parsed == full_size or (batch+1) % verbose == 0):
                print(f'{batch+1:8d}' + 
                      f'   {D.mean().item():12.4f}' +
                      f'   {H.mean().item():12.4f}' +
                      f'   {deviation.mean().item():12.4f}' +
                      f'   {P.mean().item():12.4f}' +
                      (not optimizer is None) * f'   {loss.item():18.4f}' +
                      f'    [{parsed:>7d}/{full_size:>7d}]')
        
    return _D.mean(), _H.mean(), _P.mean(), _D.std(), _H.std()

# main training function
def mtg_train(working_sample, opt_parameters, model = None, monotone = False, verbose = False):
    # global device
    
    # check inputs
    n, num_cols = working_sample.shape
    if monotone:
        d = int((num_cols - 3) / 2)
    else:
        d = int((num_cols - 2) / 3)
        
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
    
    # model creation (or recycling)
    lr =1e-4
    hidden_size = 64
    n_hidden_layers = 2
    if model is None:
        if monotone:
            phi_list = nn.ModuleList([PotentialF(1, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size)])
            psi_list = nn.ModuleList([PotentialF(1, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d)])
            h_list   = nn.ModuleList([PotentialF(1, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d)])
        else:
            phi_list = nn.ModuleList([PotentialF(1, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d)])
            psi_list = nn.ModuleList([PotentialF(1, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d)])
            h_list   = nn.ModuleList([PotentialF(d, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d)])
        model = nn.ModuleList([phi_list, psi_list, h_list])
    model = model.to(device)
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
        D, H, P, ds, hs = mtg_train_loop(model, working_loader, beta, beta_multiplier, gamma, optimizer, verb)
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
    
def mtg_dual_value(model, working_sample, opt_parameters, normalize_pi = False):
    # global device
    if 'penalization' in opt_parameters.keys() and opt_parameters['penalization'] != 'L2':
        print('penalization not implemented: ' + opt_parameters['penalization'])
        return
    beta_prime   = beta_L2_prime             # first derivative of L2 penalization function
    gamma        = opt_parameters['gamma']
    phi_list, psi_list, h_list = model
    
    working_sample = torch.tensor(working_sample, device=device).float()
    phi, psi, h, L, C, w = mtg_parse(model, working_sample)
    D = (phi + psi).sum(axis=1)   # sum over dimensions
    H = (h * L).sum(axis=1)       # sum over dimensions
    deviation = C - D - H
    pi_star = w * beta_prime(deviation, gamma)
    print(deviation.max().detach().numpy())
    sum_pi_star = pi_star.sum()
    if normalize_pi and sum_pi_star > 0:
        pi_star = pi_star / sum_pi_star
    
    return D.detach().mean().cpu().numpy(), H.detach().mean().cpu().numpy(), pi_star.detach().cpu().numpy()


# utils - construct working sample from various sources

# from a sample of (x,y)
def generate_working_sample(xy_set, cost_f, weight = None, uniform_weight = True):
    size, num_cols = xy_set.shape
    d = int(num_cols / 2)
    x = xy_set[:, :d]
    y = xy_set[:, d:]
    l = y - x          # each column i has (yi - xi)
    c = cost_f(x, y)   # vector of costs
    if weight is None:
        if uniform_weight:
            weight = np.ones(size) / size
        else:
            print('a weight must be specified')
            return None
    working_sample = np.hstack([xy_set, l, c.reshape(size, 1), weight.reshape(size, 1)])
    assert working_sample.shape[1] == 3 * d + 2
    return working_sample

# from a sample of (u,v) in the domain [0,1]^d x [0,1]^d
# u maps to x and v maps to y through the inverse cumulatives
def generate_working_sample_uv(uv_set, inv_cum_xi, inv_cum_yi, cost_f,
                               weight = None, uniform_weight = True):
    size, num_cols = uv_set.shape
    d = int(num_cols / 2)
    u = uv_set[:, :d]
    v = uv_set[:, d:]
    x = np.array([inv_cum_xi(u[:,i], i) for i in range(d)]).T
    y = np.array([inv_cum_yi(v[:,i], i) for i in range(d)]).T
    l = y - x          # each column i has (yi - xi)
    c = cost_f(x, y)   # vector of costs
    if weight is None:
        if uniform_weight:
            weight = np.ones(size) / size
        else:
            print('a weight must be specified')
            return None
    working_sample = np.hstack([uv_set, l, c.reshape(size, 1), weight.reshape(size, 1)])
    xy_set = np.hstack([x, y])
    return working_sample, xy_set

# from a sample of (u,v) in the domain [0,1] x [0,1]^d
# X is now monotone
# 1-dimension u maps to d-dimension x; d-dimension v maps to d-dimension y
def generate_working_sample_uv_mono(uv_set, inv_cum_x, inv_cum_yi, cost_f,
                                   weight = None, uniform_weight = True):
    size, num_cols = uv_set.shape
    d = int(num_cols - 1)
    u = uv_set[:, 0]       # n x 1
    v = uv_set[:, 1:d+1]   # n x d
    x = inv_cum_x(u)       # n x d
    y = np.array([inv_cum_yi(v[:,i], i) for i in range(d)]).T   # n x d
    l = y - x          # each column i has (yi - xi)
    c = cost_f(x, y)   # vector of costs
    if weight is None:
        if uniform_weight:
            weight = np.ones(size) / size
        else:
            print('a weight must be specified')
            return None
    working_sample = np.hstack([uv_set, l, c.reshape(size, 1), weight.reshape(size, 1)])
    xy_set = np.hstack([x, y])
    return working_sample, xy_set


# utils - monotonically couple a pair of discrete probabilities
def couple(X1, X2, w1, w2):
    # X1 and X2 should be sorted before calling
    # will return X, w
    #   [ X_ , w_ ] <-- single row calculated in this step
    #   [ _X , _w ] <-- stack calculated recursively
    assert len(X1) == len(w1) and len(X2) == len(w2), 'weight length error'
    assert np.isclose(sum(w1), sum(w2))
    if len(X1) == 1 and len(X2) == 1:
        assert np.isclose(w1[0], w2[0]), 'weight matching error'
        return np.array([X1[0], X2[0]]), w1[0]
    if np.isclose(w1[0], w2[0]):
        X_ = np.array([X1[0], X2[0]]).T
        w_  = w1[0]
        _X, _w = couple(X1[1:], X2[1:], w1[1:], w2[1:])
    elif w1[0] < w2[0]:
        X_ = np.array([X1[0], X2[0]])
        w_  = w1[0]
        _w2 = w2.copy()
        _w2[0] = w2[0] - w1[0]
        _X, _w = couple(X1[1:], X2, w1[1:], _w2)
    elif w1[0] > w2[0]:
        X_ = np.array([X1[0], X2[0]])
        w_  = w2[0]
        _w1 = w1.copy()
        _w1[0] = w1[0] - w2[0]
        _X, _w = couple(X1, X2[1:], _w1, w2[1:])
    X = np.vstack([X_, _X])
    w = np.vstack([w_, _w])
    return X, w
    
# utils - random (u,v) numbers from hypercube
def random_uvset(n_points, d):
    uniform_sample = np.random.random((n_points, 2*d))
    return uniform_sample

def random_uvset_mono(n_points, d):
    uniform_sample = np.random.random((n_points, d+1))
    return uniform_sample

# utils - grid (u,v) numbers from hypercube
def grid_uvset(n, d):
    n_grid = np.array(list(itertools.product(*[list(range(n)) for i in range(2 * d)])))
    uv_set = (2 * n_grid + 1) / (2 * n)   # points in the d-hypercube
    return uv_set

def grid_uvset_mono(n, d):
    n_grid = np.array(list(itertools.product(*[list(range(n)) for i in range(d+1)])))
    uv_set = (2 * n_grid + 1) / (2 * n)   # points in the d-hypercube
    return uv_set

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
    pl.show()

# utils - convergence plots
def convergence_plot_empirical(value_series_list, labels, h_series_list=None,
                               lower_bound=None, title='Numerical value - convergence'):
    pl.figure(figsize = [7,7])   # plot in two iterations to have a clean legend
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
