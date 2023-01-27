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

# module functions to be used:
# module.cost
# module.sample
# module.plot_sample
print('distribution: ' + module.distribution)
print('cost:         ' + module.cost_label)

figsize = [12,12]
_dir = './model_dump/'

# choose:
# 2d normal
labels = ['results_max_cross_product_y_normal_independent',
          'results_max_cross_product_y_normal_positive',
          'results_max_cross_product_y_normal_direct' ]

# uniform max
# labels = ...

# uniform min
# labels = ...

# adjustments
# for label in labels:
#     file = label + '.pickle'
#     _path = _dir + file
#     with open(_path, 'rb') as file:
#         results = pickle.load(file)
#     print('model loaded from ' + _path)
    
#     # adjustments
#     # results.keys()
#     # print(f'reference value {results["ref_value"]:8.4f}')
#     # results['ref_value'] = ref_value
#     # del results['f_label']
#     # del results['cost']
#     # results['cost_label'] = module.cost_label
#     # ...
#     # ---
    
#     with open(_path, 'wb') as file:
#         pickle.dump(results, file)
#     print('model saved to ' + _path)


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
    primal_obj     = results['primal_obj']
    distribution   = results['distribution']
    cost_label     = results['cost_label']
    coupling       = results['coupling']
    phi_x_list     = results['phi_x_list']
    phi_y_list     = results['phi_y_list']
    h_list         = results['h_list']
    value_series   = results['value_series']
    std_series     = results['std_series']
    penalty_series = results['penalty_series']
    ref_value      = results['ref_value']
    
    print('primal objective:  ' + primal_obj)
    print('distribution:      ' + distribution)
    print('cost function:     ' + cost_label)
    print('coupling:          ' + coupling)
    print(f'reference value:   {ref_value:8.4f}')
    
    # new sample
    n_points = 100000
    clip_normal = 4
    sample_mu_X, sample_mu_Y = module.sample(n_points)
    sample_th_X, sample_th_Y = module.sample(n_points, coupling=coupling)
    
    # plot samples
    if False:
        module.plot_sample(sample_mu_X, sample_mu_Y, 'mu')
        module.plot_sample(sample_th_X, sample_th_Y, 'th')
    
    # data loader
    mu_loader, th_loader = mmot.generate_loaders(sample_mu_X, sample_mu_Y, sample_th_X, sample_th_Y, module.batch_size)

    # single call
    print()
    print('single call')
    print('-------------------------------------------------------')
    value, std, penalty = mmot.train_loop(module.cost, module.primal_obj,
                                          mu_loader, th_loader, 
                                          phi_x_list, phi_y_list, h_list, mmot.beta_L2, module.gamma,
                                          optimizer = None, verbose = True)

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

    _, __, _th_X, _th_Y = mmot.generate_tensors(sample_mu_X, sample_mu_Y, sample_th_X, sample_th_Y)
    
    sample_size = len(_th_X)
    th_X1 = _th_X[:, 0].view(sample_size, 1)
    th_X2 = _th_X[:, 1].view(sample_size, 1)
    th_Y1 = _th_Y[:, 0].view(sample_size, 1)
    th_Y2 = _th_Y[:, 1].view(sample_size, 1)
    
    # potential functions
    phi_x1 = phi_x_list[0]
    phi_x2 = phi_x_list[1]
    phi_y1 = phi_y_list[0]
    phi_y2 = phi_y_list[1]
    h1     = h_list[0]
    h2     = h_list[1]
    
    # apply pi_hat logic to theta
    sign = 1 if primal_obj == 'max' else -1
    f_th = module.cost(_th_X, _th_Y).view(sample_size, 1)
    phi_th = phi_x1(th_X1) + phi_x2(th_X2) + phi_y1(th_Y1) + phi_y2(th_Y2)
    h_th = h1(torch.cat([th_X1, th_X2]).view(sample_size, 2)) * (th_Y1 - th_X1) + \
           h2(torch.cat([th_X1, th_X2]).view(sample_size, 2)) * (th_Y2 - th_X2)
    b_prime = mmot.beta_L2_prime(sign * (f_th - (phi_th + h_th)), gamma=module.gamma)
    b_prime = b_prime.detach().numpy()[:,0]
    
    # draw points from theta and discard according to beta'(f - h)
    select_size = 100000
    draw_probability = b_prime / b_prime.sum()
    draw_probability.max()
    select_points = np.random.choice(range(sample_size), size=select_size, p=draw_probability)
    selection = np.array([th_X1[select_points].detach().numpy()[:,0],
                          th_X2[select_points].detach().numpy()[:,0],
                          th_Y1[select_points].detach().numpy()[:,0],
                          th_Y2[select_points].detach().numpy()[:,0]]).T
    
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
    pl.plot(t, phi_x1(torch.tensor(t).float().view(len(t), 1)).detach().numpy())
    pl.plot(t, phi_x2(torch.tensor(t).float().view(len(t), 1)).detach().numpy())
    pl.legend(['phi_1(x1)', 'phi_2(x2)'])
    
    pl.figure(figsize=figsize)
    pl.xlabel('y1, y2')
    pl.ylabel('psi')
    pl.plot(t, phi_y1(torch.tensor(t).float().view(len(t), 1)).detach().numpy())
    pl.plot(t, phi_y2(torch.tensor(t).float().view(len(t), 1)).detach().numpy())
    pl.legend(['psi_1(y1)', 'psi_2(y2)'])
    
    x1x2 = torch.tensor(np.vstack([t,t]).T).float().view(len(t), 2)
    pl.figure(figsize=figsize)
    pl.xlabel('x1 (= x2)')
    pl.ylabel('h')
    pl.title('h')
    pl.plot(t, h1(x1x2).detach().numpy())
    pl.plot(t, h2(x1x2).detach().numpy())
    pl.legend(['h1(x1, x2)', 'h2(x1, x2)'])   
    