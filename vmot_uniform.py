# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:35:25 2021
@author: souzam
PyTorch implementation of Eckstein and Kupper 2019 - Computation of Optimal Transport...
"""

import mmot_dual_nn as mmot
import numpy as np
import matplotlib.pyplot as pl

import torch
from torch import nn
import pickle

import datetime as dt, time

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# Each example of MMOT numeric dual solution is defined by
#    1. a sample "mu" drown from the independent coupling of the marginals
#    2. a sample "th" from some sampling measure theta satisfying that a solution is absolutely continuous wrt theta
#    3. a cost function
# 
# To use the framework
#    a. instantiate a module of class mmot.Phi for each x marginal 
#    b. instantiate a module of class mmot.Phi for each y marginal 
#    c. instantiate a module of class mmot.Phi for each h
#    d. wrap the modules and create optimizers pointing to their parameters
#    e. create loaders for the samples
#    e. define the primal objective and the penalty function
#    f. call mmot.train_loop iteratively (this will calbireate the phi's, defining the numeric potential functions)
#    g. store the phi modules as .pickle files


# --- sampling ---
uniform_length = [2.0, 2.0, 4.0, 4.0]
def sample(n_points, coupling = ['independent', 'independent'], fix_x = None, seed = None):   # coupling in ['independent', 'positive', 'negative']
        
    # mu
    X1 = np.random.random(size=n_points) * uniform_length[0] - 0.5 * uniform_length[0]
    X2 = np.random.random(size=n_points) * uniform_length[1] - 0.5 * uniform_length[1]
    Y1 = np.random.random(size=n_points) * uniform_length[2] - 0.5 * uniform_length[2]
    Y2 = np.random.random(size=n_points) * uniform_length[3] - 0.5 * uniform_length[3]
        
    # empirical coupling (allignment)
    if coupling[0] == 'positive':
        X1 = np.sort(X1)
        X2 = np.sort(X2)
        X = np.vstack((np.sort(X1), np.sort(X2))).T
        np.random.shuffle(X)
        X1, X2 = X[:,0], X[:,1]
    if coupling[0] == 'negative':
        X1 = np.sort(X1)
        X2 = np.sort(X2)
        X = np.vstack((np.sort(X1), np.sort(X2)[::-1])).T
        np.random.shuffle(X)
        X1, X2 = X[:,0], X[:,1]
        
    if coupling[1] == 'positive':
        Y1 = np.sort(Y1)
        Y2 = np.sort(Y2)
        Y = np.vstack((np.sort(Y1), np.sort(Y2))).T
        np.random.shuffle(Y)
        Y1, Y2 = Y[:,0], Y[:,1]
    if coupling[1] == 'negative':
        Y1 = np.sort(Y1)
        Y2 = np.sort(Y2)
        Y = np.vstack((np.sort(Y1), np.sort(Y2)[::-1])).T
        np.random.shuffle(Y)
        Y1, Y2 = Y[:,0], Y[:,1]
        
    X, Y = np.array([X1, X2]).T, np.array([Y1, Y2]).T
    return X, Y
    
figsize = [12,12]
def plot_sample(X, Y, label):
    X1, X2, Y1, Y2 = X[:,0], X[:,1], Y[:,0], Y[:,1]
    pl.figure(figsize=figsize)
    pl.title(label)
    pl.axis('equal')
    pl.xlabel('X Y 1')
    pl.ylabel('X,Y 2')
    pl.scatter(Y1, Y2, alpha=.05)
    pl.scatter(X1, X2, alpha=.05)
    pl.legend(['X sample', 'Y sample'])


# --- cost function ---
# portfolio option
ax1 = 1.0
ax2 = 2.0
ay1 = 1.0
ay2 = 2.0
Kx  = 1.0
Ky  = 1.0

def f_portfolio_option(x, y):
    x1 = x[:, 0]
    x2 = x[:, 1]
    y1 = y[:, 0]
    y2 = y[:, 1]
    nominal_fx  = ax1 * x1 + ax2 * x2 - Kx
    nominal_fy  = ay1 * y1 + ay2 * y2 - Ky
    return np.maximum(nominal_fx, 0) + np.maximum(nominal_fy, 0)

def minus_f_portfolio_option(x, y):
    return -f_portfolio_option(x, y)

cost, cost_label = f_portfolio_option, 'portfolio_option'
minus_cost, minus_cost_label = minus_f_portfolio_option, 'minus_portfolio_option'
 
# --- optimization setting ---

n_points = 100000
distribution = 'uniform'
primal_obj = 'max'

# cost function and reference value
ref_value = 11 / 8                   # portfolio_option
ref_value = -1 / 8                   # minus_portfolio_option

# coupling mu1-mu2 and nu1-nu2
# coupling = ['independent', 'independent']
# coupling = ['positive', 'independent']
# coupling = ['negative', 'independent']
# coupling = ['positive', 'positive']
# coupling = ['negative', 'negative']

batch_size = 1000
gamma = 1000


# for coupling in [['independent', 'independent'],
#                   ['positive', 'independent'],
#                   ['positive', 'positive']       ]:
for coupling in [['independent', 'independent'],
                  ['negative', 'independent'],
                  ['negative', 'negative']       ]:
    # construct samples
    sample_mu_X, sample_mu_Y = sample(n_points, coupling=['independent', 'independent'], seed=1)
    sample_th_X, sample_th_Y = sample(n_points, coupling=coupling)
    if False:
        plot_sample(sample_mu_X, sample_mu_Y, 'mu')
        plot_sample(sample_th_X, sample_th_Y, 'th')
        
    # wrap samples in tensor loaders
    mu_loader, th_loader = mmot.generate_loaders(sample_mu_X, sample_mu_Y, sample_th_X, sample_th_Y, batch_size)
    
    # --- new model ---
    d = 2
    hidden_size = 32
    n_hidden_layers = 2
    phi_x_list = nn.ModuleList([mmot.Phi(1, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d)])
    phi_y_list = nn.ModuleList([mmot.Phi(1, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d)])
    h_list     = nn.ModuleList([mmot.Phi(d, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d)])
    
    # --- training: calls to train_loop ---
    print()
    print('first call')
    print('-------------------------------------------------------')
    value, std, penalty = mmot.train_loop(cost, primal_obj, mu_loader, th_loader, 
                                          phi_x_list, phi_y_list, h_list, mmot.beta_L2, gamma,
                                          optimizer = None, verbose = True)
    
    print(f'value:               {value:7.4f}')
    print(f'standard deviation:  {std:7.4f}')
    print(f'penalty:             {penalty:7.4f}')
        
    # iterative calls
    epochs = 200
    t0 = time.time() # timer
    lr =1e-4
    optimizer = torch.optim.Adam(h_list.parameters(), lr=lr)
    _value = []
    _std = []
    _penalty = []
    lr =1e-4
    print()
    print('iterative calls')
    print(f'learning rate:       {lr:0.7f}')
    print('-------------------------------------------------------')
    optimizer = torch.optim.Adam(list(phi_x_list.parameters()) + list(phi_y_list.parameters()) + list(h_list.parameters()), lr=lr)
    for t in range(epochs):
        verb = ((t == 0) | ((t+1) % 10 == 0) | ((t+1) == epochs))
        if verb:
            print()
            print(f'epoch {t+1}')
            print(f'gamma:               {gamma:0d}')
            print('-------------------------------------------------------')
        else:
            print(f'epoch {t+1}...')
        value, std, penalty = mmot.train_loop(cost, primal_obj, mu_loader, th_loader, 
                                              phi_x_list, phi_y_list, h_list, mmot.beta_L2, gamma,
                                              optimizer = optimizer, verbose = verb)
        _value.append(value)
        _std.append(std)
        _penalty.append(penalty)
        if verb:
            print()
            
    print(f'value:               {value:7.4f}')
    print(f'standard deviation:  {std:7.4f}')
    print(f'penalty:             {penalty:7.4f}')
        
    t1 = time.time() # timer
    print('duration = ' + str(dt.timedelta(seconds=round(t1 - t0))))
        
    value_series = np.array(_value)
    std_series = np.array(_std)
    penalty_series = np.array(_penalty)
    
    # show
    pl.plot(value_series)
    pl.axhline(ref_value, linestyle=':', color='black')    
    
    # summarize results
    results = { 'primal_obj'    : primal_obj,
                'distribution'  : distribution,
                'coupling'      : [coupling[0], coupling[1]],
                'cost_label'    : cost_label,
                'ref_value'     : ref_value,
                'phi_x_list'    : phi_x_list,
                'phi_y_list'    : phi_y_list,
                'h_list'        : h_list,
                'value_series'  : value_series,
                'std_series'    : std_series,
                'penalty_series': penalty_series  }
    
    # dump
    _dir = '/model_dump/'
    _file = 'results_' + primal_obj + '_' + cost_label + '_' + distribution + '_' + coupling[0] + '_' + coupling[1] + f'_{epochs}.pickle'
    # _file = 'results_' + primal_obj + '_' + f_label + '_' + distribution + '_' + coupling[0] + '_' + coupling[1] + '.pickle'
    _path = _dir + _file
    with open(_path, 'wb') as file:
        pickle.dump(results, file)
    print('model saved to ' + _path)


# load
_dir = './model_dump/'

# max portfolio_option
labels = ['results_max_portfolio_option_uniform_independent_independent',
          'results_max_portfolio_option_uniform_positive_independent',
          'results_max_portfolio_option_uniform_positive_positive' ]

# min portfolio_option
labels = ['results_max_minus_portfolio_option_uniform_independent_independent',
          'results_max_minus_portfolio_option_uniform_negative_independent',
          'results_max_minus_portfolio_option_uniform_negative_negative' ]

# show convergence results, normal marginals, cross product
pl.figure(figsize=figsize)   # plot in two iterations to have a clean legend
for label in labels:
    file = label + '.pickle'
    _path = _dir + file
    with open(_path, 'rb') as file: 
        results = pickle.load(file)
        print('model loaded from ' + _path)
    value_series   = results['value_series']
    # value_series   = results['value_series'] + results['penalty_series']
    pl.plot(value_series)
pl.legend([l[8:] for l in labels])
for label in labels:
    file = label + '.pickle'
    _path = _dir + file
    with open(_path, 'rb') as file:
        results = pickle.load(file)
        print('model loaded from ' + _path)
    value_series   = results['value_series']
    # value_series   = results['value_series'] + results['penalty_series']
    # top = results['value_series'].max()
    # bot = results['value_series'].min()
    # pl.ylim(bot - .1 * (top-bot), top + .1 * (top-bot))
    std_series     = results['std_series']
    pl.fill_between(range(len(value_series)), value_series + std_series, value_series - std_series, alpha = .5, facecolor = 'grey')
if not results['ref_value']  is None:
    pl.axhline(results['ref_value'], linestyle=':', color='black')  

# adjustments
for label in labels:
    file = label + '.pickle'
    _path = _dir + file
    with open(_path, 'rb') as file:
        results = pickle.load(file)
    print('model loaded from ' + _path)
    print(results.keys())
    # del results['f_label']
    # del results['cost']
    # results['cost_label'] = cost_label
    # results['coupling'] = [results['x_coupling'], results['y_coupling']]
    # del results['x_coupling']
    # del results['y_coupling']
    # ...
    # print(results['ref_value'])
    # results['ref_value'] = ref_value
    # ---
    
    with open(_path, 'wb') as file:
        pickle.dump(results, file)
    print('model saved to ' + _path)

# show (TO DO: clean up and use proper script)
for label in labels:
    print()
    print(label)
    file = label + '.pickle'
    _path = _dir + file
    with open(_path, 'rb') as file:
        results = pickle.load(file)
        print('model loaded from ' + _path)
        
    primal_obj     = results['primal_obj']
    distribution   = results['distribution']
    coupling       = results['coupling']
    f_label        = results['cost_label']
    gamma          = results['gamma']
    ref_value      = results['ref_value']
    phi_x_list     = results['phi_x_list']
    phi_y_list     = results['phi_y_list']
    h_list         = results['h_list']
    value_series   = results['value_series']
    std_series     = results['std_series']
    penalty_series = results['penalty_series']
    
    print('primal objective:  ' + primal_obj)
    print('function:          ' + f_label)
    print('distribution:      ' + distribution)
    print('coupling:          ' + str(coupling))
    print(f'ref value:         {ref_value:7.4f}')
    
    # new sample
    n_points = 100000
    sample_mu_X, sample_mu_Y = sample(n_points, seed=1)
    sample_th_X, sample_th_Y = sample(n_points, coupling=coupling)
    if False:
        plot_sample(sample_mu_X, sample_mu_Y, 'mu')
        plot_sample(sample_th_X, sample_th_Y, 'th')
        
    # wrap samples in tensor loaders
    mu_loader, th_loader = mmot.generate_loaders(sample_mu_X, sample_mu_Y, sample_th_X, sample_th_Y, batch_size)

    
    # single call
    print()
    print('single call')
    print('-------------------------------------------------------')
    value, std, penalty = mmot.train_loop(cost, primal_obj, mu_loader, th_loader, 
                                          phi_x_list, phi_y_list, h_list, mmot.beta_L2, gamma,
                                          optimizer = None, verbose = True)

    print(f'value:               {value:7.4f}')
    print(f'standard deviation:  {std:7.4f}')
    print(f'penalty:             {penalty:7.4f}')
    
    _mu_X = torch.tensor(sample_mu_X).float()
    _mu_Y = torch.tensor(sample_mu_Y).float()
    _th_X = torch.tensor(sample_th_X).float()
    _th_Y = torch.tensor(sample_th_Y).float()
    
    # graph sample
    full_size = len(_th_X)
    th_X1 = _th_X[:, 0].view(full_size, 1)
    th_X2 = _th_X[:, 1].view(full_size, 1)
    th_Y1 = _th_Y[:, 0].view(full_size, 1)
    th_Y2 = _th_Y[:, 1].view(full_size, 1)
    
    # potential functions
    phi_x1 = phi_x_list[0]
    phi_x2 = phi_x_list[1]
    phi_y1 = phi_y_list[0]
    phi_y2 = phi_y_list[1]
    h1     = h_list[0]
    h2     = h_list[1]
    
    # apply pi_hat logic to theta
    sign = 1 if primal_obj == 'max' else -1
    f_th = cost(_th_X, _th_Y).view(full_size, 1)
    phi_th = phi_x1(th_X1) + phi_x2(th_X2) + phi_y1(th_Y1) + phi_y2(th_Y2)
    h_th = h1(torch.cat([th_X1, th_X2]).view(full_size, 2)) * (th_Y1 - th_X1) + \
           h2(torch.cat([th_X1, th_X2]).view(full_size, 2)) * (th_Y2 - th_X2)
    b_prime = mmot.beta_L2_prime(sign * (f_th - (phi_th + h_th)), gamma=gamma)
    b_prime = b_prime.detach().numpy()[:,0]
    
    # draw points from theta and discard according to beta'(f - h)
    select_size = 100000
    draw_probability = b_prime / b_prime.sum()
    draw_probability.max()
    select_points = np.random.choice(range(len(th_X1)), size=select_size, p=draw_probability)
    selection = np.array([th_X1[select_points].detach().numpy()[:,0],
                          th_X2[select_points].detach().numpy()[:,0],
                          th_Y1[select_points].detach().numpy()[:,0],
                          th_Y2[select_points].detach().numpy()[:,0]]).T
    
    # pi_hat
    pl.figure(figsize=figsize)
    pl.axis('equal')
    pl.xlabel('X, Y 1')
    pl.ylabel('X, Y 2')
    # pl.scatter(selection[:,2], selection[:,3], alpha=.05)
    pl.scatter(selection[:,0], selection[:,1], alpha=.05)
    
    # check potential functions individually
    npoints = 201
    
    t = np.linspace(-5, 5, npoints)
    pl.figure(figsize=figsize)
    pl.xlabel('x1, x2')
    pl.ylabel('phi')
    pl.plot(t, h_x1(torch.tensor(t).float().view(len(t), 1)).detach().numpy())
    pl.plot(t, h_x2(torch.tensor(t).float().view(len(t), 1)).detach().numpy())
    pl.legend(['phi_1(x1)', 'phi_2(x2)'])
    
    pl.figure(figsize=figsize)
    pl.xlabel('y1, y2')
    pl.ylabel('psi')
    pl.plot(t, h_y1(torch.tensor(t).float().view(len(t), 1)).detach().numpy())
    pl.plot(t, h_y2(torch.tensor(t).float().view(len(t), 1)).detach().numpy())
    pl.legend(['psi_1(y1)', 'psi_2(y2)'])
    
    x1x2 = torch.tensor(np.vstack([t,t]).T).float().view(len(t), 2)
    pl.figure(figsize=figsize)
    pl.xlabel('x1 (= x2)')
    pl.ylabel('h')
    pl.title('h')
    pl.plot(t, g1(x1x2).detach().numpy())
    pl.plot(t, g2(x1x2).detach().numpy())
    pl.legend(['h1(x1, x2)', 'h2(x1, x2)'])