# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:35:25 2021
@author: souzam
PyTorch implementation of Eckstein and Kupper 2019 - Computation of Optimal Transport...
"""

import mmot_dual_nn as mmot
import numpy as np
import matplotlib.pyplot as pl

import scipy.stats as stats
import torch
from torch import nn
from torch.utils.data import DataLoader
import pickle

import datetime as dt, time

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'   # pytorch version issues


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


# Simple example with d = 3, normal marginals clipped at 4 std, cross-product cost(x,y) = b12.y1.y2 + b13.y1.y3  + b23.y2.y3

# --- sampling ---
# normal marginals with the standard deviations below
x_scale   = [1.0, 2.0, 2.0, 3.0, 3.0]
y_scale   = [2.0, 3.0, 4.0, 5.0, 6.0]
def sample(n_points, coupling = 'independent', clip_normal = None, fix_x = None, seed = None):
    # coupling in ['independent', 'positive', 'exact']
    if not seed is None:
        np.random.seed(seed)
        
    if clip_normal is None:   # noisy tails included
        if coupling == 'straight':
            # special case, theoretical straight coupling
            # ** pending **
            print('3d straight coupling not implemented')
            return None
        else:
            if fix_x is None:
                X1 = np.random.normal(loc=0.0, scale=x_scale[0], size=n_points)
                X2 = np.random.normal(loc=0.0, scale=x_scale[1], size=n_points)
                X3 = np.random.normal(loc=0.0, scale=x_scale[2], size=n_points)
                X4 = np.random.normal(loc=0.0, scale=x_scale[3], size=n_points)
                X5 = np.random.normal(loc=0.0, scale=x_scale[4], size=n_points)
            else:
                X1 = np.repeat(fix_x[0], n_points)
                X2 = np.repeat(fix_x[1], n_points)
                X3 = np.repeat(fix_x[2], n_points)
                X4 = np.repeat(fix_x[3], n_points)
                X5 = np.repeat(fix_x[4], n_points)
            Y1 = np.random.normal(loc=0.0, scale=y_scale[0], size=n_points)
            Y2 = np.random.normal(loc=0.0, scale=y_scale[1], size=n_points)
            Y3 = np.random.normal(loc=0.0, scale=y_scale[2], size=n_points)
            Y4 = np.random.normal(loc=0.0, scale=y_scale[3], size=n_points)
            Y5 = np.random.normal(loc=0.0, scale=y_scale[4], size=n_points)
        
    else:   # clip tails
        # mu
        rv = stats.truncnorm(-clip_normal, clip_normal)
        if coupling == 'exact':
            # special case, theoretical straight coupling
            # ** pending **
            print('3d straight coupling not implemented')
            return None
        else:
            if fix_x is None:
                random_sample = rv.rvs(size=[n_points,5])
                X = random_sample * x_scale
                X1, X2, X3, X4, X5 = X[:,0], X[:,1], X[:,2], X[:,3], X[:,4]
            else:
                X1 = np.repeat(fix_x[0], n_points)
                X2 = np.repeat(fix_x[1], n_points)
                X3 = np.repeat(fix_x[2], n_points)
                X4 = np.repeat(fix_x[3], n_points)
                X5 = np.repeat(fix_x[4], n_points)
            random_sample = rv.rvs(size=[n_points,5])
            Y = random_sample * y_scale
            Y1, Y2, Y3, Y4, Y5 = Y[:,0], Y[:,1], Y[:,2], Y[:,3], Y[:,4]
    
    # empirical coupling (allignment)
    if coupling == 'positive':
        X1 = np.sort(X1)
        X2 = np.sort(X2)
        X3 = np.sort(X3)
        X4 = np.sort(X4)
        X5 = np.sort(X5)
        X = np.vstack((np.sort(X1), np.sort(X2), np.sort(X3), np.sort(X4), np.sort(X5))).T
        np.random.shuffle(X)
        X1, X2, X3, X4, X5 = X[:,0], X[:,1], X[:,2], X[:,3], X[:,4]
    
    X, Y = np.array([X1, X2, X3, X4, X5]).T, np.array([Y1, Y2, Y3, Y4, Y5]).T
    return X, Y

# generate samples
clip_normal = None
n_points = 100000
sample_mu_X, sample_mu_Y = sample(n_points, coupling='independent', clip_normal=clip_normal, seed=1)
sample_th_X, sample_th_Y = sample(n_points, coupling='positive', clip_normal=clip_normal)

# wrap samples in tensor loaders
batch_size = 1000
shuffle = True
_mu_X = torch.tensor(sample_mu_X).float()
_mu_Y = torch.tensor(sample_mu_Y).float()
_th_X = torch.tensor(sample_th_X).float()
_th_Y = torch.tensor(sample_th_Y).float()
mu_dataset = mmot.SampleDataset(_mu_X, _mu_Y)
th_dataset = mmot.SampleDataset(_th_X, _th_Y)
mu_loader = DataLoader(mu_dataset, batch_size = batch_size, shuffle = shuffle)
th_loader = DataLoader(th_dataset, batch_size = batch_size, shuffle = shuffle)

'''
# test mode only
X = sample_mu_X
Y = sample_mu_Y
'''

# --- cost function ---
# y only
label_cross_product_y    = 'cross_product_y_R5'
def cost_cross_product_y(x, y):
    y1 = y[:, 0]
    y2 = y[:, 1]
    y3 = y[:, 2]
    y4 = y[:, 3]
    y5 = y[:, 4]
    return y1 * y2 + y1 * y3 + y1 * y4 + y1 * y5 + y2 * y3 + y2 * y4 + y2 * y5 + y3 * y4 + y3 * y5 + y4 * y5

# penalty function
beta = mmot.beta_L2

# ref value
sig1, sig2, sig3, sig4, sig5 = x_scale
rho1, rho2, rho3, rho4, rho5 = y_scale
lamb1 = np.sqrt(rho1 ** 2 - sig1 ** 2)
lamb2 = np.sqrt(rho2 ** 2 - sig2 ** 2)
lamb3 = np.sqrt(rho3 ** 2 - sig3 ** 2)
lamb4 = np.sqrt(rho4 ** 2 - sig4 ** 2)
lamb5 = np.sqrt(rho5 ** 2 - sig5 ** 2)

ref_max = (sig1 * sig2 + lamb1 * lamb2) + \
          (sig1 * sig3 + lamb1 * lamb3) + \
          (sig1 * sig4 + lamb1 * lamb4) + \
          (sig1 * sig5 + lamb1 * lamb5) + \
          (sig2 * sig3 + lamb2 * lamb3) + \
          (sig2 * sig4 + lamb2 * lamb4) + \
          (sig2 * sig5 + lamb2 * lamb5) + \
          (sig3 * sig4 + lamb3 * lamb4) + \
          (sig3 * sig5 + lamb3 * lamb5) + \
          (sig4 * sig5 + lamb4 * lamb5)


# --- optimization ---

# parameters (choose below)

primal_obj = 'max'

# clip_normal = None
clip_normal = 4

# cost function
cost_function, f_label = cost_cross_product_y, label_cross_product_y 

# coupling = 'independent'
# coupling = 'positive'
# coupling = 'exact'

for coupling in ['independent', 'positive']:
    # --- new model (resets weights) ---
    d = 5
    hidden_size = 32
    n_hidden_layers = 2
    phi_x_list = nn.ModuleList([mmot.Phi(1, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d)])
    phi_y_list = nn.ModuleList([mmot.Phi(1, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d)])
    h_list     = nn.ModuleList([mmot.Phi(d, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d)])
    
    gamma = 1000
    epochs = 2000
    
    
    # --- training - calls to train_loop ---
    _value = []
    _std = []
    _penalty = []
    
    lr =1e-4
    optimizer = torch.optim.Adam(h_list.parameters(), lr=lr)
    print('\nfirst call')
    print(f'learning rate:       {lr:0.7f}')
    print('-------------------------------------------------------')
    value, std, penalty = mmot.train_loop(cost_function, primal_obj, mu_loader, th_loader, 
                                          phi_x_list, phi_y_list, h_list, beta, gamma,
                                          optimizer = optimizer, verbose = True)
    _value.append(value)
    _std.append(std)
    _penalty.append(penalty)
    print(f'value:               {value:7.4f}')
    print(f'standard deviation:  {std:7.4f}')
    print(f'penalty:             {penalty:7.4f}')
    
    # iterative calls
    t0 = time.time() # timer
    lr =1e-4
    print()
    print('iterative calls')
    print(f'learning rate:       {lr:0.7f}')
    print('-------------------------------------------------------')
    optimizer = torch.optim.Adam(list(phi_x_list.parameters()) + list(phi_y_list.parameters()) + list(h_list.parameters()), lr=lr)
    for t in range(epochs):
        verb = ((t == 0) | ((t+1) % 100 == 0) | ((t+1) == epochs))
        if verb:
            print()
            print(f'epoch {t+1}')
            print(f'gamma:               {gamma:0d}')
            print('-------------------------------------------------------')
        else:
            print(f'epoch {t+1}...')
        value, std, penalty = mmot.train_loop(cost_function, primal_obj, mu_loader, th_loader, 
                                              phi_x_list, phi_y_list, h_list, beta, gamma,
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
    
    # summarize results
    results = { 'distribution'  : 'normal',
                'f_label'       : f_label,
                'primal_obj'    : primal_obj,
                'coupling'      : coupling,
                'phi_x_list'    : phi_x_list,
                'phi_y_list'    : phi_y_list,
                'h_list'        : h_list,
                'value_series'  : value_series,
                'std_series'    : std_series,
                'penalty_series': penalty_series  }
    
    # dump
    _dir = 'U:/Projects/MMOT/'
    _file = 'results_' + primal_obj + '_' + f_label + '_' + coupling + f'_{d}d.pickle'
    _path = _dir + _file
    with open(_path, 'wb') as file:
        pickle.dump(results, file)
    print('model saved to ' + _path)

# show
pl.fill_between(range(len(value_series)), value_series + std_series, value_series - std_series, alpha = .5, facecolor = 'grey')
pl.plot(value_series)
pl.axhline(ref_max, linestyle=':', color='black')    


# load & show
_dir = 'U:/Projects/MMOT/'
labels = ['results_max_cross_product_y_independent_5d',
          'results_max_cross_product_y_positive_5d' ]
files = [l + '.pickle' for l in labels]


normal_scale   = [2.0, 1.0, 3.0, 4.0]
l1 = np.sqrt(normal_scale[2] ** 2 - normal_scale[0] ** 2)
l2 = np.sqrt(normal_scale[3] ** 2 - normal_scale[1] ** 2)



# show convergence results, normal marginals, cross product
figsize = [12, 12]
cut = 1500 + 1
pl.figure(figsize=figsize)   # plot in two iterations to have a clean legend
for file in files:    
    _path = _dir + file
    with open(_path, 'rb') as file:
        results = pickle.load(file)
        print('model loaded from ' + _path)
    value_series   = results['value_series'][:cut]
    # value_series   = results['value_series'] + results['penalty_series']
    pl.plot(value_series)
pl.legend([l[8:] for l in labels])
for file in files:    
    _path = _dir + file
    with open(_path, 'rb') as file:
        results = pickle.load(file)
        print('model loaded from ' + _path)
    value_series   = results['value_series'][:cut]
    # value_series   = results['value_series'] + results['penalty_series']
    # top = results['value_series'].max()
    # bot = results['value_series'].min()
    # pl.ylim(bot - .1 * (top-bot), top + .1 * (top-bot))
    std_series     = results['std_series'][:cut]
    pl.fill_between(range(len(value_series)), value_series + std_series, value_series - std_series, alpha = .5, facecolor = 'grey')
if not ref_max  is None:
    pl.axhline(ref_max, linestyle=':', color='black')  
    
    
# choose file
# label = labels[0]
for label in labels:
    print()
    print(label)
    file = label + '.pickle'
    _path = _dir + file
    with open(_path, 'rb') as file:
        results = pickle.load(file)
        print('model loaded from ' + _path)
    distribution   = results['distribution']
    f_label        = results['f_label']
    primal_obj     = results['primal_obj']
    efficient      = results['efficient']
    h_list         = results['h_list']
    value_series   = results['value_series']
    std_series     = results['std_series']
    penalty_series = results['penalty_series']
    cost = cost_function[f_label]
    
    print('distribution:      ' + distribution)
    print('function:          ' + f_label)
    print('primal objective:  ' + primal_obj)
    print('efficient?         ' + ('yes' if efficient else 'no'))
    
    # # plot value series (check convergence)
    # pl.figure(figsize=figsize)
    # pl.title(label[8:] + ', value')
    # pl.plot(value_series)
    # if not ref_value  is None:
    #     pl.axhline(ref_value, linestyle=':', color='black')  
    
    # pl.figure(figsize=figsize)
    # pl.title(label[8:] + ', (value - penalty)')
    # top = value_series.max()
    # bot = value_series.min()
    # pl.ylim(bot - .1 * (top-bot), top + .1 * (top-bot))
    # pl.plot(value_series - penalty_series)
    # if not ref_value  is None:
    #     pl.axhline(ref_value, linestyle=':', color='black')  
    
    # new sample
    n_points = 100000
    clip_normal = 4
    if efficient:
        theta_x_coupling = 'positive' if primal_obj == 'max' else 'negative'
    else:
        theta_x_coupling = 'independent'
    _mu, _th = sample(n_points, distribution=distribution, theta_x_coupling=theta_x_coupling, clip_normal=clip_normal, seed=1)
    mu = torch.tensor(np.array([_mu[:,0], _mu[:,1], _mu[:,2], _mu[:,3]]).T).float()
    th = torch.tensor(np.array([_th[:,0], _th[:,1], _th[:,2], _th[:,3]]).T).float()
    
    torch.mean(cost(mu))
    torch.mean(cost(th))
    
    # plot samples
    if False:
        plot_sample(_mu, 'mu', fix_axes=False)
        plot_sample(_th, 'th', fix_axes=False)
    
    # data loader
    batch_size = 1000
    shuffle = True
    dataset = SampleDataset(mu, th)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
    
    # single call
    gamma = 1000
    print('single call')
    value, std, penalty = train_loop(loader, cost, h_list, beta, primal_obj, gamma, verbose=True)
    print(f'value:               {value:7.4f}')
    print(f'standard deviation:  {std:7.4f}')
    print(f'penalty:             {penalty:7.4f}')
    
    # graph sample
    fix_x = False
    if fix_x:
        # tentative pi_x
        fix_x = sample_single_x(distribution=distribution)
        _, _th = sample(n_points, distribution=distribution, theta_x_coupling=theta_x_coupling, theta_fix_x = fix_x, clip_normal = 3)
        th = torch.tensor(np.array([_th[:,0], _th[:,1], _th[:,2], _th[:,3]]).T).float()
    
    if True:
        plot_sample(_th, 'th', fix_axes=False)
    
    full_size = len(th)
    th_X1 = th[:, 0].view(full_size, 1)
    th_X2 = th[:, 1].view(full_size, 1)
    th_Y1 = th[:, 2].view(full_size, 1)
    th_Y2 = th[:, 3].view(full_size, 1)
    
    # potential functions
    h_x1 = h_list[0]
    h_x2 = h_list[1]
    h_y1 = h_list[2]
    h_y2 = h_list[3]
    g1   = h_list[4]
    g2   = h_list[5]
    
    # apply pi_hat logic to theta
    sign = 1 if primal_obj == 'max' else -1
    f_th = cost(th).view(full_size, 1)
    h_th = h_x1(th_X1) + h_x2(th_X2) + h_y1(th_Y1) + h_y2(th_Y2)
    g_th = g1(torch.cat([th_X1, th_X2]).view(full_size, 2)) * (th_Y1 - th_X1) + \
           g2(torch.cat([th_X1, th_X2]).view(full_size, 2)) * (th_Y2 - th_X2)
    b_prime = beta_L2_prime(sign * (f_th - (h_th + g_th)), gamma=gamma)
    b_prime = b_prime.detach().numpy()[:,0]
    
    # draw points from theta and discard according to beta'(f - h)
    select_size = 100000
    draw_probability = b_prime / b_prime.sum()
    draw_probability.max()
    select_points = np.random.choice(range(len(th)), size=select_size, p=draw_probability)
    selection = th[select_points].detach().numpy()
    
    # pi_hat
    pl.figure(figsize=figsize)
    pl.axis('equal')
    pl.xlabel('X, Y 1')
    pl.ylabel('X, Y 2')
    pl.scatter(selection[:,2], selection[:,3], alpha=.05)
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
    