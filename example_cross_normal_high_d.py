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
print('A', A)
print('B', B)

<<<<<<< Updated upstream
for d in [2, 3, 4, 5]:
    
    # iterations
    I = 10
    existing_i = 0   # new
    n_points = 1000000
=======
E_series = []
for d in [2, 3, 4, 5, 6, 7, 8]:
    
    # iterations
    I = 10
    existing_i = 10   # new
    n_points = 2000000
>>>>>>> Stashed changes
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
        print(f'\nmodel {existing_i} loaded')
        model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.load_results(f'normal_d{d}_{existing_i}')
        model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.load_results(f'normal_mono_d{d}_{existing_i}')
    
    # plot
<<<<<<< Updated upstream
    evo1 = np.array(D_evo1) # random, independent
    evo2 = np.array(D_evo2) # random, monotone
    # h1 = np.array(H_evo1)   # random, independent
    # h2 = np.array(H_evo2)   # random, monotone
    vmot.convergence_plot([evo2, evo1], ['monotone', 'independent'], ref_value=ref_value)
    # vmot.convergence_plot([evo2, evo1], ['monotone', 'independent'], h_series_list=[h2, h1], ref_value=ref_value)
=======
    length = 50
    evo1 = np.array(D_evo1)[:length] # random, independent
    evo2 = np.array(D_evo2)[:length] # random, monotone
    E_series.append([d, evo1, evo2, ref_value])
    vmot.convergence_plot([evo2, evo1], ['monotone', 'independent'], ref_value=ref_value, title=f'Convergence - empirical marginals (d = {d})')
    # h1 = np.array(H_evo1)[:length]   # random, independent
    # h2 = np.array(H_evo2)[:length]   # random, monotone
    # vmot.convergence_plot([evo2, evo1], ['monotone', 'independent'], h_series_list=[h2, h1], ref_value=ref_value, title=f'Convergence - empirical marginals (d = {d})')
>>>>>>> Stashed changes
    
    # iterate
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
<<<<<<< Updated upstream
        vmot.convergence_plot([evo2, evo1], ['monotone', 'independent'], ref_value=ref_value)
=======
        vmot.convergence_plot([evo2, evo1], ['monotone', 'independent'], ref_value=ref_value)
    
        h1 = np.array(H_evo1)   # random, independent
        h2 = np.array(H_evo2)   # random, monotone
        vmot.convergence_plot([evo2, evo1], ['monotone', 'independent'], h_series_list=[h2, h1], ref_value=ref_value)




# multiple convergence plots
ref_color='black'
title='Convergence - normal'
fig, ax = pl.subplots(2, 2, figsize = [12,12])   # plot in two iterations to have a clean legend
for i, E in enumerate(E_series[:4]):
    _ax = ax.flatten()[i]
    d, evo1, evo2, ref_value = E
    _ax.plot(range(1, len(evo1)+1), evo1)
    _ax.plot(range(1, len(evo2)+1), evo2)
    _ax.axhline(ref_value, linestyle=':', color=ref_color)
    _ax.set_title(f'd = {d}')
ax[0][1].legend(['full', 'reduced'])
fig.suptitle('Convergence - normal marginals')
pl.show()




















>>>>>>> Stashed changes
