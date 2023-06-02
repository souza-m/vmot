# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:35:25 2021
@author: souzam
PyTorch implementation of Eckstein and Kupper 2019 - Computation of Optimal Transport...
"""

# Example 1.2: solve cross-product cost with normal marginals, d > 2


import numpy as np
from scipy.stats import norm
import vmot

# processing parameters
d = 10

for d in [5, 6, 7, 8, 10]:
    
    # iterations
    I = 20
    existing_i = -1   # new
    if d == 5:
        existing_i = 11
        I = 1
    if d == 6:
        existing_i = 11
        I = 1
    if d == 7:
        existing_i = 21
        I = 1
    if d == 8:
        existing_i = 21
        I = 1
    if d == 10:
        existing_i = 14
        I = 6
    
    n_points = 2000000
    print(f'd: {d}')
    print(f'sample size: {n_points}')
    opt_parameters = { 'penalization'    : 'L2',
                       'beta_multiplier' : 1,
                       'gamma'           : 100,
                       'batch_size'      : 2000,   # no special formula for this
                       'epochs'          : 10     }
    
    # cost function to be minimized
    A = np.empty((d, d)) * np.nan
    B = np.empty((d, d)) * np.nan
    for i in range(0, d):
        for j in range(i+1, d):
            A[i,j] = 0.0
            B[i,j] = 1.0
    B[0, 1] = 10.0
    
    def cost_f(x, y):
        cost = 0.0
        for i in range(0, d):
            for j in range(i+1, d):
                cost = cost + A[i,j] * x[:,i] * x[:,j] + B[i,j] * y[:,i] * y[:,j]
        return cost
    
    # sets of (u,v) points
    uvset1 = vmot.random_uvset(n_points, d)
    uvset2 = vmot.random_uvset_mono(n_points, d)
    print('sample shapes')
    print('independent random  ', uvset1.shape)
    print('monotone random     ', uvset2.shape)
    
    
    # choose scales
    sig = np.ones(d)                 # x normal distribution scales (std)
    rho = np.sqrt(2.0) * np.ones(d)  # y normal distribution scales (std)
    
    # reference value (see proposition)
    lam = np.sqrt(rho ** 2 - sig ** 2)
    ref_value = 0.0
    for i in range(0, d):
        for j in range(i+1, d):
            ref_value = ref_value + (A[i,j] + B[i,j]) * sig[i] * sig[j] + B[i,j] * lam[i] * lam[j]
    print(f'normal marginals exact solution: {ref_value:8.4f}')
    
    # inverse cumulatives
    def normal_inv_cum_xi(q, i):
        return norm.ppf(q) * sig[i]
    
    def normal_inv_cum_yi(q, i):
        return norm.ppf(q) * rho[i]
    
    def normal_inv_cum_x(q):
        z = norm.ppf(q)
        return np.array([z * sig[i] for i in range(d)]).T
    
    if existing_i < 0:
        
        # new random sample
        uvset1 = vmot.random_uvset(n_points, d)
        uvset2 = vmot.random_uvset_mono(n_points, d)
        ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, normal_inv_cum_xi, normal_inv_cum_yi, cost_f)
        ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, normal_inv_cum_x, normal_inv_cum_yi, cost_f)
        
        # train/store
        model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.mtg_train(ws1, opt_parameters, monotone = False, verbose = 10)
        model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.mtg_train(ws2, opt_parameters, monotone = True, verbose = 10)
        vmot.dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1], f'normal_{d}_{i}')
        vmot.dump_results([model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2], f'normal_mono_{d}_{i}')
        existing_i = 0
    
    else:
        # load
        model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.load_results(f'normal_{d}_{existing_i}')
        model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.load_results(f'normal_mono_{d}_{existing_i}')
    
    # plot
    evo1 = np.array(D_evo1) # random, independent
    evo2 = np.array(D_evo2) # random, monotone
    h1 = np.array(H_evo1)   # random, independent
    h2 = np.array(H_evo2)   # random, monotone
    vmot.convergence_plot([evo2, evo1], ['monotone', 'independent'], ref_value=ref_value)
    vmot.convergence_plot([evo2, evo1], ['monotone', 'independent'], h_series_list=[h2, h1], ref_value=ref_value)
    
    
    # iterate
    for i in range(existing_i+1, existing_i+1+I):
        
        # new random sample
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
        
        vmot.dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1], f'normal_{d}_{i}')
        vmot.dump_results([model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2], f'normal_mono_{d}_{i}')
        
        # plot
        evo1 = np.array(D_evo1) # random, independent
        evo2 = np.array(D_evo2) # random, monotone
        vmot.convergence_plot([evo2, evo1], ['monotone', 'independent'], ref_value=ref_value)
    
        h1 = np.array(H_evo1)   # random, independent
        h2 = np.array(H_evo2)   # random, monotone
        vmot.convergence_plot([evo2, evo1], ['monotone', 'independent'], h_series_list=[h2, h1], ref_value=ref_value)
