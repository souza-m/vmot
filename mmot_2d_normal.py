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
def sample(n_points, coupling = 'independent', clip_normal = 4, fix_x = None, seed = None):   # coupling in ['independent', 'positive', 'negative', 'straight']
    
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
a = 0
b = 1
def f_cross_product(x, y):
    x1 = x[:, 0]
    x2 = x[:, 1]
    y1 = y[:, 0]
    y2 = y[:, 1]
    return a * x1 * x2 + b * y1 * y2

cost = f_cross_product
cost_label    = 'cross_product'
distribution = 'normal'

# reference value
sig1 = normal_scale[0]
sig2 = normal_scale[1]
rho1 = normal_scale[2]
rho2 = normal_scale[3]
l1 = np.sqrt(rho1 ** 2 - sig1 ** 2)
l2 = np.sqrt(rho2 ** 2 - sig2 ** 2)
ref_value = 2 + l1 * l2            # max cross_product
# ref_value = None                   # unknown

# optimization parameters
d = 2
primal_obj = 'max'
batch_size = 1000
gamma = 1000
clip_normal = 4
# clip_normal = None

if __name__ == "__main__":

    n_points = 100000
    epochs = 2000
    
    for coupling in ['independent', 'positive', 'direct']:
        
        # construct samples
        sample_mu_X, sample_mu_Y = sample(n_points, coupling='independent', clip_normal=clip_normal, seed=1)
        sample_th_X, sample_th_Y = sample(n_points, coupling=coupling, clip_normal=clip_normal)
        if False:
            plot_sample(sample_mu_X, sample_mu_Y, 'mu')
            plot_sample(sample_th_X, sample_th_Y, 'th')
            
        # wrap samples in tensor loaders
        mu_loader, th_loader = mmot.generate_loaders(sample_mu_X, sample_mu_Y, sample_th_X, sample_th_Y, batch_size)
        
        # --- new model ---
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
