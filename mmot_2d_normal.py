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
 
normal_scale   = [2.0, 1.0, 3.0, 4.0]
def sample(n_points, coupling = 'independent', clip_normal = None, fix_x = None, seed = None):   # coupling in ['independent', 'positive', 'negative', 'straight']
    
    if not seed is None:
        np.random.seed(seed)
        
    if clip_normal is None:   # noisy tails included
        
        if coupling == 'direct':
            # *** special case, theoretical straight coupling ***
            # fix_x not available here
            normal_a = np.random.normal(loc=0.0, scale=1.0, size=n_points)
            normal_b = np.random.normal(loc=0.0, scale=1.0, size=n_points)
            l1 = np.sqrt(normal_scale[2] ** 2 - normal_scale[0] ** 2)
            l2 = np.sqrt(normal_scale[3] ** 2 - normal_scale[1] ** 2)
            X1 = normal_a * normal_scale[0]
            X2 = normal_a * normal_scale[1]
            Y1 = X1 + normal_b * l1
            Y2 = X2 + normal_b * l2
        else:
            if fix_x is None:
                X1 = np.random.normal(loc=0.0, scale=normal_scale[0], size=n_points)
                X2 = np.random.normal(loc=0.0, scale=normal_scale[1], size=n_points)
                Y1 = np.random.normal(loc=0.0, scale=normal_scale[2], size=n_points)
                Y2 = np.random.normal(loc=0.0, scale=normal_scale[3], size=n_points)
            else:
                X1 = np.repeat(fix_x[0], n_points)
                X2 = np.repeat(fix_x[1], n_points)
                Y1 = np.random.normal(loc=0.0, scale=normal_scale[2], size=n_points)
                Y2 = np.random.normal(loc=0.0, scale=normal_scale[3], size=n_points)
        
    else:   # clip tails
        # mu
        rv = stats.truncnorm(-clip_normal, clip_normal)
        if coupling == 'direct':
            # *** special case, theoretical straight coupling of normal marginals ***
            random_sample = rv.rvs(size=[n_points,2])
            normal_a, normal_b = random_sample[:,0], random_sample[:,1]
            l1 = np.sqrt(normal_scale[2] ** 2 - normal_scale[0] ** 2)
            l2 = np.sqrt(normal_scale[3] ** 2 - normal_scale[1] ** 2)
            X1 = normal_a * normal_scale[0]
            X2 = normal_a * normal_scale[1]
            Y1 = X1 + normal_b * l1
            Y2 = X2 + normal_b * l2
        else:
            if fix_x is None:
                random_sample = rv.rvs(size=[n_points,4])
                XY = random_sample * normal_scale
                X1, X2, Y1, Y2 = XY[:,0], XY[:,1], XY[:,2], XY[:,3]
            else:
                X1 = np.repeat(fix_x[0], n_points)
                X2 = np.repeat(fix_x[1], n_points)
                random_sample = rv.rvs(size=[n_points,2])
                Y = random_sample * normal_scale
                Y1, Y2 = Y[:,0], Y[:,1]
    
    # empirical coupling (allignment)
    if coupling == 'positive':
        X1 = np.sort(X1)
        X2 = np.sort(X2)
        X = np.vstack((np.sort(X1), np.sort(X2))).T
        np.random.shuffle(X)
        X1, X2 = X[:,0], X[:,1]
    if coupling == 'negative':
        X1 = np.sort(X1)
        X2 = np.sort(X2)
        X = np.vstack((np.sort(X1), np.sort(X2)[::-1])).T
        np.random.shuffle(X)
        X1, X2 = X[:,0], X[:,1]
    
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
# cross product
a = 2
b = 3
def f_cross_product(x, y):
    x1 = x[:, 0]
    x2 = x[:, 1]
    y1 = y[:, 0]
    y2 = y[:, 1]
    return a * x1 * x2 + b * y1 * y2

# cross product, y only
def f_cross_product_y(x, y):
    y1 = y[:, 0]
    y2 = y[:, 1]
    return y1 * y2

f_label_cross_product    = 'cross_product'
f_label_cross_product_y  = 'cross_product_y'
cost_function = { f_label_cross_product   : f_cross_product,
                  f_label_cross_product_y : f_cross_product_y   }

# reference value
sig1 = normal_scale[0]
sig2 = normal_scale[1]
rho1 = normal_scale[2]
rho2 = normal_scale[3]
l1 = np.sqrt(rho1 ** 2 - sig1 ** 2)
l2 = np.sqrt(rho2 ** 2 - sig2 ** 2)
ref_value = 2 + l1 * l2            # max cross_product_y
# ref_value = None                   # unknown
 
# --- optimization setting ---

n_points = 100000
distribution = 'normal'

# choose parameters below

# cost and penalty functions
# f_label = f_label_cross_product
f_label = f_label_cross_product_y
cost = cost_function[f_label]

# clip_normal = None
clip_normal = 4

# objective
primal_obj = 'max'

# normal
# coupling = 'independent'
# coupling = 'positive'
# coupling = 'direct'

gamma = 1000
    
for coupling in ['independent', 'positive', 'direct']:
    
    # construct samples
    sample_mu_X, sample_mu_Y = sample(n_points, coupling='independent', clip_normal=clip_normal, seed=1)
    sample_th_X, sample_th_Y = sample(n_points, coupling=coupling, clip_normal=clip_normal)
    if False:
        plot_sample(sample_mu_X, sample_mu_Y, 'mu')
        plot_sample(sample_th_X, sample_th_Y, 'th')
        
    # wrap samples in tensor loaders
    batch_size = 1000
    # shuffle = True
    # _mu_X = torch.tensor(sample_mu_X).float()
    # _mu_Y = torch.tensor(sample_mu_Y).float()
    # _th_X = torch.tensor(sample_th_X).float()
    # _th_Y = torch.tensor(sample_th_Y).float()
    # mu_dataset = mmot.SampleDataset(_mu_X, _mu_Y)
    # th_dataset = mmot.SampleDataset(_th_X, _th_Y)
    # mu_loader = DataLoader(mu_dataset, batch_size = batch_size, shuffle = shuffle)
    # th_loader = DataLoader(th_dataset, batch_size = batch_size, shuffle = shuffle)
    mu_loader, th_loader = mmot.generate_loaders(sample_mu_X, sample_mu_Y, sample_th_X, sample_th_Y, batch_size)
    
    # --- new model ---
    d = 2
    hidden_size = 32
    n_hidden_layers = 2
    phi_x_list = nn.ModuleList([mmot.Phi(1, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d)])
    phi_y_list = nn.ModuleList([mmot.Phi(1, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d)])
    h_list     = nn.ModuleList([mmot.Phi(d, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d)])
    
    # --- training: calls to train_loop ---
    lr =1e-4
    optimizer = torch.optim.Adam(h_list.parameters(), lr=lr)
    print()
    print('first call')
    print(f'learning rate:       {lr:0.7f}')
    print('-------------------------------------------------------')
    value, std, penalty = mmot.train_loop(cost, primal_obj, mu_loader, th_loader, 
                                          phi_x_list, phi_y_list, h_list, mmot.beta_L2, gamma,
                                          optimizer = optimizer, verbose = True)
    print(f'value:               {value:7.4f}')
    print(f'standard deviation:  {std:7.4f}')
    print(f'penalty:             {penalty:7.4f}')
    
    # iterative calls
    epochs = 2000
    t0 = time.time() # timer
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
    ref_value = 10 + 15 * np.sqrt(3)
    
    # show
    pl.plot(value_series)
    pl.axhline(ref_value, linestyle=':', color='black') 
    
    # summarize results
    results = { 'primal_obj'    : primal_obj,
                'f_label'       : f_label,
                'cost'          : cost,
                'distribution'  : distribution,
                'clip_normal'   : clip_normal,
                'ref_value'     : ref_value,
                'gamma'         : gamma,
                'coupling'      : coupling,
                'phi_x_list'    : phi_x_list,
                'phi_y_list'    : phi_y_list,
                'h_list'        : h_list,
                'value_series'  : value_series,
                'std_series'    : std_series,
                'penalty_series': penalty_series  }

    # dump
    _dir = '/model_dump/'
    # _file = 'results_' + primal_obj + '_' + f_label + '_' + distribution + '_' + coupling + f'_{epochs}.pickle'
    _file = 'results_' + primal_obj + '_' + f_label + '_' + distribution + '_' + coupling + '.pickle'
    _path = _dir + _file
    with open(_path, 'wb') as file:
        pickle.dump(results, file)
    print('model saved to ' + _path)


# load
_dir = '/model_dump/'

labels = ['results_max_cross_product_y_normal_independent',
          'results_max_cross_product_y_normal_positive',
          'results_max_cross_product_y_normal_direct' ]
          
files = [l + '.pickle' for l in labels]


# adjustments
label = 'results_max_cross_product_normal_independent_2d'
label = 'results_max_cross_product_normal_positive_2d'
label = 'results_max_cross_product_normal_direct_2d'
print(label)
file = label + '.pickle'
_path = _dir + file
with open(_path, 'rb') as file:
    results = pickle.load(file)
# adjustments
results.keys()
# results['coupling'] = 'independent'
# results['coupling'] = 'positive'
# results['coupling'] = 'direct'
with open(_path, 'wb') as file:
    pickle.dump(results, file)

# adjustments
for label in labels:
    print(label)
    _dir = '/model_dump/'
    _file = label + '.pickle'
    _path = _dir + _file
    with open(_path, 'rb') as file:
        results = pickle.load(file)
    print('model loaded from ' + _path)
    
    # adjustments
    # ...
    print(f'ref_value {results["ref_value"]:8.4f}')
    # results['ref_value'] = ref_value
    # ---
    
    _dir = './model_dump/'
    _path = _dir + _file
    with open(_path, 'wb') as file:
        pickle.dump(results, file)
    print('model saved to ' + _path)


# show convergence results, normal marginals, cross product
pl.figure(figsize=figsize)   # plot in two iterations to have a clean legend
for file in files:    
    _path = _dir + file
    with open(_path, 'rb') as file:
        results = pickle.load(file)
        print('model loaded from ' + _path)
    value_series   = results['value_series']
    # value_series   = results['value_series'] + results['penalty_series']
    pl.plot(value_series)
pl.legend([l[8:] for l in labels])
for file in files:    
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
if not ref_value  is None:
    pl.axhline(ref_value, linestyle=':', color='black')  
    
    
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
    coupling       = results['coupling']
    h_list         = results['h_list']
    value_series   = results['value_series']
    std_series     = results['std_series']
    penalty_series = results['penalty_series']
    cost = cost_function[f_label]
    
    print('distribution:      ' + distribution)
    print('function:          ' + f_label)
    print('primal objective:  ' + primal_obj)
    print('coupling:          ' + coupling)
    
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
    sample_mu_X, sample_mu_Y = sample(n_points, coupling='independent', clip_normal=clip_normal, seed=1)
    sample_th_X, sample_th_Y = sample(n_points, coupling=coupling)
    # plot samples
    if False:
        plot_sample(sample_mu_X, sample_mu_Y, 'mu')
        plot_sample(sample_th_X, sample_th_Y, 'th')
    
    # data loader
    batch_size = 1000
    shuffle = True
    # _mu_X = torch.tensor(sample_mu_X).float()
    # _mu_Y = torch.tensor(sample_mu_Y).float()
    # _th_X = torch.tensor(sample_th_X).float()
    # _th_Y = torch.tensor(sample_th_Y).float()
    # mu_dataset = mmot.SampleDataset(_mu_X, _mu_Y)
    # th_dataset = mmot.SampleDataset(_th_X, _th_Y)
    # mu_loader = DataLoader(mu_dataset, batch_size = batch_size, shuffle = shuffle)
    # th_loader = DataLoader(th_dataset, batch_size = batch_size, shuffle = shuffle)
    mu_loader, th_loader = mmot.generate_loaders(sample_mu_X, sample_mu_Y, sample_th_X, sample_th_Y)

    # single call
    gamma = 1000   # to do: read from file
    print('single call')
    value, std, penalty = train_loop(loader, cost, h_list, beta, primal_obj, gamma, verbose=True)
    print(f'value:               {value:7.4f}')
    print(f'standard deviation:  {std:7.4f}')
    print(f'penalty:             {penalty:7.4f}')
    
    # _mu_X = torch.tensor(sample_mu_X).float()
    # _mu_Y = torch.tensor(sample_mu_Y).float()
    # _th_X = torch.tensor(sample_th_X).float()
    # _th_Y = torch.tensor(sample_th_Y).float()
    
    # graph sample
    # fix_x = False
    # if fix_x:
    #     # tentative pi_x
    #     fix_x = sample_single_x(distribution=distribution)
    #     _, _th = sample(n_points, distribution=distribution, theta_x_coupling=theta_x_coupling, theta_fix_x = fix_x, clip_normal = 3)
    #     th = torch.tensor(np.array([_th[:,0], _th[:,1], _th[:,2], _th[:,3]]).T).float()

    
    if False:
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
    