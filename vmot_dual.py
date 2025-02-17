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
# import itertools

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
    - u, v, w which are used as inputs to the neural networks
    - x, y, z which are points in the problem domain mapped through inverse cumulatives
    - differences from x to y and from y to z
    - c = cost
    - w = weight (associated with the sample element, typically 1/n)

2. Create a new "model" -- set of potential functions phi and h implemented as neural networks
    
2. Call mtg_optimize, which in turn calls the train_loop to optimize the neural networks
'''

# penalty function and its derivative
# def beta_Lp(x, p, gamma):
#     return (1 / gamma) * (1 / p) * torch.pow(torch.relu(gamma * x), p)
 
# def beta_L2(x, gamma):
#     return beta_Lp(x, 2, gamma)
 
# def beta_L2_prime(x, gamma):
#     return (1 / gamma) * torch.relu(x)

# def beta_Lp_conj(y, p):
#     q = p / (p - 1)
#     return y ** q / q 
    
# def beta_L2_conj(y):
#     return beta_Lp_conj(y, 2)
    
# beta_L2(torch.tensor(np.linspace(-5, 10, 21)), gamma=100)

# penalty function and its derivative
def beta_Lp(x, p, gamma):
    return (gamma / p) * torch.pow(torch.relu(x), p)
 
def beta_L2(x, gamma):
    return beta_Lp(x, 2, gamma)
 
def beta_L2_prime(x, gamma):
    return gamma * torch.relu(x)

def beta_Lp_conj(y, p):
    q = p / (p - 1)
    return y ** q / q 
    
def beta_L2_conj(y):
    return beta_Lp_conj(y, 2)
    

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


# main training function
def mtg_train(working_sample, model, opt, d, T, monotone, opt_parameters, verbose = 0):

    beta            = beta_L2
    gamma           = opt_parameters['gamma']
    batch_size      = opt_parameters['batch_size']
    epochs          = opt_parameters['epochs']
    
    # loader
    shuffle        = True     # must be True to avoid some bias when sample is ordered
    working_sample = torch.tensor(working_sample, device=device).float()
    working_loader = DataLoader(working_sample, batch_size = batch_size, shuffle = shuffle)
    
    # iterative calls to train_loop
    D_series = []
    H_series = []
    if verbose > 0:
        t0 = time.time() # timer
    for i in range(epochs):
        # if verbose > 0 and (i==0 or (i+1)%verbose == 0):
        print(f'epoch {i+1:4d}')
        verb = ((i+1)%verbose == 0 or (i+1 == epochs)) * 100
        if verb:
            print()
        D, H, P, ds, hs = mtg_train_loop(working_loader, model, opt, d, T, monotone, beta, gamma, verb)
        D_series.append(D)
        H_series.append(H)
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
        
    return D_series, H_series
    
# train loop
def mtg_train_loop(working_loader, model, opt, d, T, monotone, beta, gamma, verbose = 0):
    
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
        D, H, c = mtg_parse(sample, model, d, T, monotone)
        deviation = c - D - H
        penalty = beta(deviation, gamma)
        
        # loss and backpropagation
        loss = (D.sum() + H.sum() + penalty.sum()) / full_size
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


# extract data from working sample and apply model to find dual values
# working sample format when d = 2, T = 2:
#    full dimension:    u0, u1, v0, v1, dif_x1y1, dif_x2y2, c
#    reduced dimension: u0,     v0, v1, dif_x1y1, dif_x2y2, c
# working sample format when d = 2, T = 3:
#    full dimension:    u0, u1, v0, v1, w0, w1, dif_x1y1, dif_x2y2, dif_y1z1, dif_y2z2, c
#    reduced dimension: u0,     v0, v1, w0, w1, dif_x1y1, dif_x2y2, dif_y1z1, dif_y2z2, c
def mtg_parse(sample, model, d, T, monotone):
    
    phi_list, h_list = model
    size, n_cols = sample.shape
    q_size = T * d - monotone * (d - 1)   # quantile inputs
    dif_size = (T - 1) * d
    assert len(phi_list) == q_size, f'{len(phi_list)} {q_size}'
    assert len(h_list) == dif_size
    assert n_cols == q_size + dif_size + 1
    
    # extract from the working sample
    q_list = [sample[:,i] for i in range(q_size)]
    dif_list = [sample[:,q_size+i] for i in range(dif_size)]
    c = sample[:, -1]
    
    # dual value
    D = sum([phi(q[:,None]) for phi, q in zip(phi_list, q_list)])
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
    
    return D, H, c
    

# calculate the dual value from existing model
# this value should be close to the primal value for well trained models
def mtg_dual_value(working_sample, model, d, T, monotone):
    sample = torch.tensor(working_sample, device=device).float()
    D, H, c = mtg_parse(sample, model, d, T, monotone)
    return D.detach().mean().cpu().item(),\
           H.detach().mean().cpu().item(),\
           c.detach().mean().cpu().item()


# calculate the upper and lower margins to the true value based on eq. (2.5) of Th. 2.2, Eckstein and Kupper 2021
def mtg_numeric_pi_hat(working_sample, model, d, T, monotone, opt_parameters):
    beta_prime   = beta_L2_prime             # first derivative of L2 penalization function
    beta_conj    = beta_L2_conj
    gamma        = opt_parameters['gamma']
    
    sample = torch.tensor(working_sample, device=device).float()
    D, H, c = mtg_parse(sample, model, d, T, monotone)
    deviation = c - (D + H)
    ratio = beta_prime(deviation, gamma)
    size = len(sample)
    upper_margin = (1/gamma) * ((beta_conj(ratio)).sum() / size).detach().cpu().numpy()
    pi_hat = ratio / size
    return pi_hat.detach().cpu().numpy(), 0., upper_margin


# working sample (d = 2)
# full dimension:    u0, u1, v0, v1, dif_x1y1, dif_x2y2, c
# reduced dimension: u0,     v0, v1, dif_x1y1, dif_x2y2, c
def generate_working_sample_T2(u, v, x, y, c):
    size = len(u)
    dif_xy = y - x
    return np.hstack([u, v, dif_xy, c.reshape([size,1])])
    

# working sample (d = 2)
# full dimension:    u0, u1, v0, v1, w0, w1, dif_x1y1, dif_x2y2, dif_y1z1, dif_y2z2, c
# reduced dimension: u0,     v0, v1, w0, w1, dif_x1y1, dif_x2y2, dif_y1z1, dif_y2z2, c
def generate_working_sample_T3(u, v, w, x, y, z, c):
    size = len(u)
    dif_xy = y - x
    dif_yz = z - y
    return np.hstack([u, v, w, dif_xy, dif_yz, c.reshape([size,1])])
    

# model generation
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


# utils - file dump
# to do: use pytorch serialization scheme instead of pickle
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


# def dump_results(model, opt, D_series, d, T, monotone, label = 'test'):
#     label = label + f'_model_d{d}_T{T}_' + 'monotone' if montone else 'full'
    
#     cpu_device = torch.device('cpu')
#     for i in range(len(results)):
#         if isinstance(results[i], torch.nn.modules.container.ModuleList):
#             # print(i, type(results[i]))
#             results[i] = results[i].to(cpu_device)
    
#     # dump
#     _path = _dir + _file_prefix + label + _file_suffix
#     with open(_path, 'wb') as file:
#         pickle.dump(results, file)
#     print('model saved to ' + _path)


# utils - convergence plots
def convergence_plot(value_series_list, labels, h_series_list=None,
                     ref_value=None, title='Numerical value - convergence'):
    ref_color='black'
    ref_label='reference'
    pl.figure(figsize = [7,7])   # plot in two iterations to have a clean legend
    for v in value_series_list:
        pl.plot(range(101, len(v)+101), v)
    if not ref_value is None:
        pl.axhline(ref_value, linestyle=':', color=ref_color)
    pl.legend(labels + [ref_label])
    for v in value_series_list:
        pl.scatter(range(101, len(v)+101), v)
    if not h_series_list is None:
        pl.gca().set_prop_cycle(None)   # reset color cycler
        for v, h in zip(value_series_list, h_series_list):
            pl.plot(range(1, len(v)+1), v+h, linestyle=':')
    # pl.xlabel('epoch')
    pl.title(title)
    pl.annotate('lower bound', (len(v)*3/4, ref_value-500), color='darkgrey')   # trial and error to find a good position
    pl.show()
