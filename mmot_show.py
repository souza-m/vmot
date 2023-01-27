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
import pickle

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# choose
# import mmot_uniform as module
import mmot_2d_normal as module

# will be used:
module.cost
module.cost_label
module.sample

print('cost label ' + module.cost_label)

_dir = './model_dump/'
figsize = [12,12]

# choose:

# 2d normal
labels = ['results_max_cross_product_y_normal_independent',
          'results_max_cross_product_y_normal_positive',
          'results_max_cross_product_y_normal_direct' ]

# uniform max
# labels = ...

# uniform min
# labels = ...

# show convergence results, normal marginals, cross product
pl.figure(figsize=figsize)   # plot in two iterations to have a clean legend
for label in labels:
    file = label + '.pickle'
    _path = _dir + file
    with open(_path, 'rb') as file:
        results = pickle.load(file)
        print('model loaded from ' + _path)
    value_series   = results['value_series'] # + results['penalty_series']
    pl.plot(value_series)
pl.legend([l[8:] for l in labels])
for label in labels:
    file = label + '.pickle'
    _path = _dir + file
    with open(_path, 'rb') as file:
        results = pickle.load(file)
        print('model loaded from ' + _path)
    value_series   = results['value_series'] # + results['penalty_series']
    # top = results['value_series'].max()
    # bot = results['value_series'].min()
    # pl.ylim(bot - .1 * (top-bot), top + .1 * (top-bot))
    std_series     = results['std_series']
    pl.fill_between(range(len(value_series)), value_series + std_series, value_series - std_series, alpha = .5, facecolor = 'grey')
    if 'ref_value' in results.keys():
        pl.axhline(results['ref_value'], linestyle=':', color='black')
    
    
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
    cost           = cost_function[f_label]
    
    print('distribution:      ' + distribution)
    print('function:          ' + f_label)
    print('primal objective:  ' + primal_obj)
    print('coupling:          ' + coupling)
    
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
    