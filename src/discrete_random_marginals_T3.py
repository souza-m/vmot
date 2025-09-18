# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:35:25 2021
@author: souzam
PyTorch implementation of Eckstein and Kupper 2019 - Computation of Optimal Transport...
"""

import numpy as np
import pandas as pd
import vmot_dual as vmot
import matplotlib.pyplot as pl
import datetime as dt, time

# example with cross-product cost and discrete random marginals, d = 2, t = 3

# over all possible joint distributions of (x,y)

# two marginals 1, 2
# three periods x, y, z
d = 2
T = 3

# global variables
np.random.seed(1)
p = 2
disp = 0.3

# supports for each period
support_X = np.array([-2, -1, 0, 1, 2])
support_Y = np.array([-3, -2, -1, 0, 1, 2, 3])
support_Z = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])

# generate random mass arrays for x1, x2
# mass = 1, support = [-2, -1, 0, 1, 2], normalized to sum to 1
X1_masses = np.random.rand(len(support_X))
X1_masses = X1_masses / np.sum(X1_masses)

X2_masses = np.random.rand(len(support_X))
X2_masses = X2_masses / np.sum(X2_masses)

# generate y1, y2 by spreading x1, x2 masses
def spread_masses(input_masses, input_support, output_support, disp):
    """
    spread random percentage × disp of mass at n to n-1 and n+1, rest stays at n
    """
    output_masses = np.zeros(len(output_support))
    
    for i, n in enumerate(input_support):
        # find index of n in output_support
        n_idx = np.where(output_support == n)[0][0]
        
        # generate random percentage for spreading
        spread_pct = np.random.rand() * disp
        
        # mass to spread to neighbors
        spread_mass = input_masses[i] * spread_pct
        
        # mass that stays at n
        stay_mass = input_masses[i] * (1 - spread_pct)
        
        # add staying mass
        output_masses[n_idx] += stay_mass
        
        # spread to n-1 if it exists in support
        if n_idx > 0:
            output_masses[n_idx - 1] += spread_mass / 2
        else:
            # if n-1 doesn't exist, add back to n
            output_masses[n_idx] += spread_mass / 2
            
        # spread to n+1 if it exists in support
        if n_idx < len(output_support) - 1:
            output_masses[n_idx + 1] += spread_mass / 2
        else:
            # if n+1 doesn't exist, add back to n
            output_masses[n_idx] += spread_mass / 2
    
    # normalize to sum to 1
    output_masses = output_masses / np.sum(output_masses)
    return output_masses

# generate y1, y2 from x1, x2
Y1_masses = spread_masses(X1_masses, support_X, support_Y, disp)
Y2_masses = spread_masses(X2_masses, support_X, support_Y, disp)

# generate z1, z2 from y1, y2
Z1_masses = spread_masses(Y1_masses, support_Y, support_Z, disp)
Z2_masses = spread_masses(Y2_masses, support_Y, support_Z, disp)

# discrete inverse cumulative functions
# return the lowest n in support such that u < cumulative sum up to n
def create_discrete_F_inv(masses, support):
    """
    create discrete inverse cumulative function
    """
    cumsum = np.cumsum(masses)
    
    def F_inv(u):
        # handle both scalar and array inputs
        u = np.asarray(u)
        # find first index where cumsum >= u
        idx = np.searchsorted(cumsum, u, side='right')
        # handle edge case where u = 1.0
        idx = np.clip(idx, 0, len(support) - 1)
        return support[idx]
    
    return F_inv

F_inv_X1 = create_discrete_F_inv(X1_masses, support_X)
F_inv_X2 = create_discrete_F_inv(X2_masses, support_X)
F_inv_Y1 = create_discrete_F_inv(Y1_masses, support_Y)
F_inv_Y2 = create_discrete_F_inv(Y2_masses, support_Y)
F_inv_Z1 = create_discrete_F_inv(Z1_masses, support_Z)
F_inv_Z2 = create_discrete_F_inv(Z2_masses, support_Z)

# generate synthetic sample
# u maps to x, v maps to y, w maps to z through the inverse cumulatives
def random_sample(n_points, monotone):
    u = np.random.random((n_points, 1 if monotone else 2))
    v = np.random.random((n_points, 2))
    w = np.random.random((n_points, 2))
    
    if monotone:
        x = np.array([F_inv_X1(u[:,0]), F_inv_X2(u[:,0])]).T
    else:
        x = np.array([F_inv_X1(u[:,0]), F_inv_X2(u[:,1])]).T
    y = np.array([F_inv_Y1(v[:,0]), F_inv_Y2(v[:,1])]).T
    z = np.array([F_inv_Z1(w[:,0]), F_inv_Z2(w[:,1])]).T
                  
    return u, v, w, x, y, z

# cost function
def cost_f(x, y, z):
    return np.exp(z[:,0] + z[:,1])

# --- process batches and save models (takes long time) ---
opt_parameters = { 'gamma'           : 10,      # penalization parameter
                   'epochs'          : 1,       # iteration parameter
                   'batch_size'      : 1000  }  # iteration parameter  

if __name__ == "__main__":
    # optimization parameters
    # batch control
    I = 100              # total desired iterations
    n_points = 1000000   # sample points at each iteration
    timers = []
    for monotone in [True, False]:
        
        existing_i = 0       # last iteration saved
        mono_label = 'mono' if monotone else 'full'
        print(mono_label)
        t0 = time.time() # timer
        timers.append(t0)
        if existing_i == 0:
            # new random sample
            print('\niteration 1 (new model)\n')
            
            # regular coupling
            u, v, w, x, y, z = random_sample(n_points, monotone = monotone)
            c = cost_f(x, y, z)
            ws = vmot.generate_working_sample_T3(u, v, w, x, y, z, c)
            print('sample generated, shape ', ws.shape)
            
            # models
            model, opt = vmot.generate_model(d, T, monotone = monotone)
            print('model generated')
            
            # train
            D_series, H_series = vmot.mtg_train(ws, model, opt, d, T, monotone = monotone, opt_parameters=opt_parameters, verbose = 1)
            
            # store models and evolution
            existing_i = 1
            vmot.dump_results([model, opt, D_series], f'discrete_d{d}_T{T}_{existing_i}_' + mono_label)
            
        # iterative parsing
        while existing_i < I:
            
            # new random sample
            print(f'\niteration {existing_i+1}\n')
            
            # regular coupling
            u, v, w, x, y, z = random_sample(n_points, monotone = monotone)
            c = cost_f(x, y, z)
            ws = vmot.generate_working_sample_T3(u, v, w, x, y, z, c)
            print('sample generated, shape ', ws.shape)
            
            # load existing model
            print(f'\nloading model {existing_i}')
            model, opt, D_series = vmot.load_results(f'discrete_d{d}_T{T}_{existing_i}_' + mono_label)
            
            # train
            _D_series, _H_series = vmot.mtg_train(ws, model, opt, d, T, monotone = monotone, opt_parameters=opt_parameters, verbose = 1)
            existing_i += 1
            print('models updated')
            
            # sequential storage of the variables' evolution
            D_series = D_series + _D_series
            H_series = H_series + _H_series
            vmot.dump_results([model, opt, D_series], f'discrete_d{d}_T{T}_{existing_i}_' + mono_label)
    
        t1 = time.time() # timer
        timers.append(t1)
        print('duration = ' + str(dt.timedelta(seconds=round(t1 - t0))))