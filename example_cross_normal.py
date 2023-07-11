# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:35:25 2021
@author: souzam
PyTorch implementation of Eckstein and Kupper 2019 - Computation of Optimal Transport...
"""

# Example 1: solve cross-product cost with normal marginals, d >= 2

import numpy as np
from scipy.stats import norm
import vmot
import matplotlib.pyplot as pl
from cycler import cycler

# random parameters for the marginal distributions
np.random.seed(1)
max_d = 5
sig = np.around(np.random.random(max_d), 2) + 1
rho = np.around(np.random.random(max_d), 2) + 2
print('sig', sig)
print('rho', rho)

# random parameters for the cost functions
A = np.zeros((max_d, max_d))
B = np.around(np.random.random((max_d, max_d)), 2)
for i in range(0, max_d):
    for j in range(i+1, max_d):
        A[i,j] = 0.0
        B[i,j] = 1.0
        B[i,j] = np.around(np.random.random(), 2)
# print('A', A)
print('B', B)

# parse / load
E_series = []
# for d in [2, 3, 4, 5]:   
for d in [2]:   # for heat map only
    
    # iterations
    I = 10
    # existing_i = 0   # new
    existing_i = 10
    n_points = 1000000
    print()
    print(f'd = {d}')
    print(f'sample size: {n_points}')
    opt_parameters = { 'penalization'    : 'L2',   # fixed
                       'beta_multiplier' : 1,
                       'gamma'           : 100,
                       'batch_size'      : 2000,   # no special formula for this
                       'epochs'          : 10     }
    
    # cost function to be minimized
    def cost_f(x, y):
        cost = 0.0
        for i in range(0, d):
            for j in range(i+1, d):
                cost = cost + A[i,j] * x[:,i] * x[:,j] + B[i,j] * y[:,i] * y[:,j]
        return cost
    
    # sets of (u,v) points
    uvset1 = vmot.random_uvset(n_points, d)
    uvset2 = vmot.random_uvset_mono(n_points, d)
    print('sample shapes:')
    print('full dimension  ', uvset1.shape)
    print('reduced random  ', uvset2.shape)
    
    # reference value (see formula in proposition)
    lam = np.sqrt(rho ** 2 - sig ** 2)
    ref_value = 0.0
    for i in range(0, d):
        for j in range(i+1, d):
            ref_value = ref_value + (A[i,j] + B[i,j]) * sig[i] * sig[j] + B[i,j] * lam[i] * lam[j]
    print(f'exact solution: {ref_value:8.4f}')
    
    # inverse cumulatives
    def normal_inv_cum_xi(q, i):
        return norm.ppf(q) * sig[i]
    
    def normal_inv_cum_yi(q, i):
        return norm.ppf(q) * rho[i]
    
    def normal_inv_cum_x(q):
        z = norm.ppf(q)
        return np.array([z * sig[i] for i in range(d)]).T
    
    if existing_i == 0:
        # new random sample
        print('\niteration 1 (new model)\n')
        uvset1 = vmot.random_uvset(n_points, d)
        uvset2 = vmot.random_uvset_mono(n_points, d)
        ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, normal_inv_cum_xi, normal_inv_cum_yi, cost_f)
        ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, normal_inv_cum_x, normal_inv_cum_yi, cost_f)
        
        # train/store
        model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.mtg_train(ws1, opt_parameters, monotone = False, verbose = 10)
        model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.mtg_train(ws2, opt_parameters, monotone = True, verbose = 10)
        existing_i = 1
        print('models generated')
        vmot.dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1], f'normal_d{d}_{existing_i}')
        vmot.dump_results([model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2], f'normal_mono_d{d}_{existing_i}')
    
    else:
        # load
        print(f'\nloading model {existing_i}')
        model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.load_results(f'normal_d{d}_{existing_i}')
        model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.load_results(f'normal_mono_d{d}_{existing_i}')
    
    # iterate optimization
    while existing_i < I:
        
        # new random sample
        print(f'\niteration {existing_i+1}\n')
        uvset1 = vmot.random_uvset(n_points, d)
        uvset2 = vmot.random_uvset_mono(n_points, d)
        ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, normal_inv_cum_xi, normal_inv_cum_yi, cost_f)
        ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, normal_inv_cum_x, normal_inv_cum_yi, cost_f)
        
        _model1, _D_evo1, _H_evo1, _P_evo1, _ds_evo1, _hs_evo1 = vmot.mtg_train(ws1, opt_parameters, model=model1, monotone = False, verbose = 10)
        _model2, _D_evo2, _H_evo2, _P_evo2, _ds_evo2, _hs_evo2 = vmot.mtg_train(ws2, opt_parameters, model=model2, monotone = True, verbose = 10)
        
        D_evo1  = D_evo1  + _D_evo1
        H_evo1  = H_evo1  + _H_evo1
        P_evo1  = P_evo1  + _P_evo1
        ds_evo1 = ds_evo1 + _ds_evo1
        hs_evo1 = hs_evo1 + _hs_evo1
        model1 = _model1
        
        D_evo2  = D_evo2  + _D_evo2
        H_evo2  = H_evo2  + _H_evo2
        P_evo2  = P_evo2  + _P_evo2
        ds_evo2 = ds_evo2 + _ds_evo2
        hs_evo2 = hs_evo2 + _hs_evo2
        model2 = _model2
        
        existing_i = existing_i + 1
        print('models updated')
        vmot.dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1], f'normal_d{d}_{existing_i}')
        vmot.dump_results([model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2], f'normal_mono_d{d}_{existing_i}')
        
        # plot
        evo1 = np.array(D_evo1) # random, independent
        evo2 = np.array(D_evo2) # random, monotone
        vmot.convergence_plot([evo2, evo1], ['reduced', 'full'], ref_value=ref_value)

    # report mean and std over a collection of samples
    report = False
    if report:
        collection_size = 10
        D1_series = []
        D2_series = []
        P1_series = []
        P2_series = []
        for i in range(collection_size):
            uvset1 = vmot.random_uvset(n_points, d)
            uvset2 = vmot.random_uvset_mono(n_points, d)
            ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, normal_inv_cum_xi, normal_inv_cum_yi, cost_f)
            ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, normal_inv_cum_x, normal_inv_cum_yi, cost_f)
            D1, P1, __ = vmot.mtg_dual_value(model1, ws1, opt_parameters, normalize_pi = False)
            D2, P2, __ = vmot.mtg_dual_value(model2, ws2, opt_parameters, normalize_pi = False)
            D1_series.append(D1)
            D2_series.append(D2)
            P1_series.append(P1)
            P2_series.append(P2)
        print('dual value')
        print(f'full:     mean = {np.mean(D1_series):8.4f};   std = {np.std(D1_series):8.4f}')
        print(f'reduced:  mean = {np.mean(D2_series):8.4f};   std = {np.std(D2_series):8.4f}')
        print('penalty')
        print(f'full:     mean = {np.mean(P1_series):8.4f};   std = {np.std(P1_series):8.4f}')
        print(f'reduced:  mean = {np.mean(P2_series):8.4f};   std = {np.std(P2_series):8.4f}')
    
    # store series for multiple plot
    length = 100
    evo1 = np.array(D_evo1)[:length] # random, independent
    evo2 = np.array(D_evo2)[:length] # random, monotone
    E_series.append([d, evo1, evo2, ref_value])
        # vmot.convergence_plot([evo2, evo1], ['reduced', 'full'], ref_value=ref_value, title=f'Convergence - empirical marginals (d = {d})')
    

# chosen color cycler (see empirical example)
cc = cycler('color', ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00', '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2'])

# multiple convergence plots
ref_color='black'
title='Convergence - normal'
fig, ax = pl.subplots(2, 2, figsize = [8,8])   # plot in two iterations to have a clean legend
for i, E in enumerate(E_series[:4]):
    _ax = ax.flatten()[i]
    _ax.set_prop_cycle(cc)
    d, evo1, evo2, ref_value = E
    _ax.plot(range(1, len(evo2)+1), evo1)
    _ax.plot(range(1, len(evo1)+1), evo2)
    _ax.axhline(ref_value, linestyle=':', color=ref_color)
    _ax.set_title(f'd = {d}')
    shift = .02 * (_ax.get_ylim()[1] - pl.gca().get_ylim()[0])
    _ax.annotate('true value', (len(evo1)*2/3, ref_value+shift), color=ref_color)   # trial and error to find a good position
ax[0][1].legend(['full', 'reduced'])
# fig.suptitle('Convergence - normal marginals')
pl.tight_layout()
pl.show()


# heat map
# step 1: run the parse / load block above with d=2
# step 2:
grid_n  = 30
uvset1g = vmot.grid_uvset(grid_n, d)
uvset2g = vmot.grid_uvset_mono(grid_n, d)

# pi star, using grid samples
ws1g, grid1 = vmot.generate_working_sample_uv(uvset1g, normal_inv_cum_xi, normal_inv_cum_yi, cost_f)
ws2g, grid2 = vmot.generate_working_sample_uv_mono(uvset2g, normal_inv_cum_x, normal_inv_cum_yi, cost_f)
D1, H1, pi_star1 = vmot.mtg_dual_value(model1, ws1g, opt_parameters, normalize_pi = False)
D2, H2, pi_star2 = vmot.mtg_dual_value(model2, ws2g, opt_parameters, normalize_pi = False)

pi_star1.sum()
pi_star2.sum()
pi_star1 = pi_star1 / pi_star1.sum()
pi_star2 = pi_star2 / pi_star2.sum()

# vmot.plot_sample_2d(ws1g, label='ws1', w=pi_star1, random_sample_size=100000)
# vmot.plot_sample_2d(ws1g, label='ws2', w=pi_star2, random_sample_size=100000)

    
heat = vmot.heatmap(grid1[:,:2], pi_star1)   # X, independent
heat = vmot.heatmap(grid2[:,:2], pi_star2)   # X, monotone
heat = vmot.heatmap(grid1[:,:2], pi_star1, uplim=np.nanmax(heat))   # X, independent, sharing color scale with monotone

# check diagonal (test)
heat_marginal = np.nansum(heat, axis=0)
fig, ax = pl.subplots()
ax.set_ylim(0, max(heat_marginal))
ax.plot(heat_marginal)

pl.figure()
pl.plot(pi_star2)
