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
import scipy.stats.norm as norm

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# --- cost function ---
cost_label    = 'cross_product'
distribution = 'normal'
a = 0
b = 1
normal_scale   = [2.0, 1.0, 3.0, 4.0]

def cost(x, y):
    x1 = x[:, 0]
    x2 = x[:, 1]
    y1 = y[:, 0]
    y2 = y[:, 1]
    return a * x1 * x2 + b * y1 * y2

def cost_hat(x_hat, y_hat):
    x1 = norm.ppf(x_hat[:, 0]) * normal_scale[0]
    x2 = norm.ppf(x_hat[:, 1]) * normal_scale[1]
    y1 = norm.ppf(y_hat[:, 0]) * normal_scale[2]
    y2 = norm.ppf(y_hat[:, 1]) * normal_scale[3]
    x, y = np.array([x1, x2]).T, np.array([y1, y2]).T
    return cost(x, y)

def L_hat(xi_hat, yi_hat):
    return norm.ppf(yi_hat) - norm.ppf(xi_hat)

# reference value
sig1 = normal_scale[0]
sig2 = normal_scale[1]
rho1 = normal_scale[2]
rho2 = normal_scale[3]
l1 = np.sqrt(rho1 ** 2 - sig1 ** 2)
l2 = np.sqrt(rho2 ** 2 - sig2 ** 2)
ref_value = 2 + l1 * l2            # max cross_product


# optimization parameters
d = 2
batch_size = 1000
gamma = 1000

if __name__ == "__main__":

    grid_n = 20
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
        value, std, penalty = mmot.train_loop(cost, mu_loader, th_loader, 
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
            value, std, penalty = mmot.train_loop(cost, mu_loader, th_loader, 
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
        results = { 'cost_label'    : cost_label,
                    'cost'          : cost,
                    'distribution'  : distribution,
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
        # _file = 'results_' + f_label + '_' + distribution + '_' + coupling + f'_{epochs}.pickle'
        _file = 'results_' + cost_label + '_' + distribution + '_' + coupling + '.pickle'
        _path = _dir + _file
        with open(_path, 'wb') as file:
            pickle.dump(results, file)
        print('model saved to ' + _path)

# --- adjustments ---
_dir = '/model_dump/'
labels = ['results_max_cross_product_y_normal_independent',
          'results_max_cross_product_y_normal_positive',
          'results_max_cross_product_y_normal_direct' ]

for label in labels:
    file = label + '.pickle'
    _path = _dir + file
    with open(_path, 'rb') as file:
        results = pickle.load(file)
    print('model loaded from ' + _path)
    
    # adjustments
    # results.keys()
    # print(f'reference value {results["ref_value"]:8.4f}')
    # results['ref_value'] = ref_value
    # del results['f_label']
    # del results['cost']
    # results['cost_label'] = module.cost_label
    # ...
    # ---
    
    with open(_path, 'wb') as file:
        pickle.dump(results, file)
    print('model saved to ' + _path)

