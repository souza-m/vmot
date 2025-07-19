# -*- coding: utf-8 -*-
"""
created on mon may 24 17:35:25 2021
@author: souzam
pytorch implementation of eckstein and kupper 2019 - computation of optimal transport...
"""

import numpy as np
import pandas as pd
import vmot_dual as vmot
import matplotlib.pyplot as pl
import datetime as dt, time


# example with cross-product cost and discrete random marginals, d = 2, T = 2
# over all possible joint distributions of (x,y)

# global variables
np.random.seed(1)
p = 2
disp = 0.3

# two marginals 1, 2
# two periods x, y
d = 2
T = 2

# support for x1, x2
X_support = np.array([-2, -1, 0, 1, 2])
# support for y1, y2
Y_support = np.array([-3, -2, -1, 0, 1, 2, 3])

# generate random mass arrays for x1, x2 (normalized to sum to 1)
X1_masses = np.random.rand(len(X_support))
X1_masses = X1_masses / np.sum(X1_masses)

X2_masses = np.random.rand(len(X_support))
X2_masses = X2_masses / np.sum(X2_masses)

# generate y1, y2 by spreading masses with dispersion
def generate_Y_masses(X_masses, X_support, Y_support, disp):
    """generate y masses by spreading x masses with dispersion"""
    Y_masses = np.zeros(len(Y_support))
    
    for i, x_val in enumerate(X_support):
        x_mass = X_masses[i]
        
        # find position of x_val in y_support
        y_idx = np.where(Y_support == x_val)[0]
        if len(y_idx) == 0:
            continue  # x_val not in y_support
        y_idx = y_idx[0]
        
        # random percentage to spread
        spread_fraction = np.random.rand() * disp
        spread_mass = x_mass * spread_fraction
        remain_mass = x_mass - spread_mass
        
        # put remaining mass at current position
        Y_masses[y_idx] += remain_mass
        
        # spread mass to neighbors
        if spread_mass > 0:
            # split spread_mass between left and right neighbors
            left_mass = spread_mass * np.random.rand()
            right_mass = spread_mass - left_mass
            
            # add to left neighbor if exists
            if y_idx > 0:
                Y_masses[y_idx - 1] += left_mass
            else:
                Y_masses[y_idx] += left_mass  # add back to current if no left neighbor
                
            # add to right neighbor if exists
            if y_idx < len(Y_support) - 1:
                Y_masses[y_idx + 1] += right_mass
            else:
                Y_masses[y_idx] += right_mass  # add back to current if no right neighbor
    
    # normalize to sum to 1
    Y_masses = Y_masses / np.sum(Y_masses)
    return Y_masses

Y1_masses = generate_Y_masses(X1_masses, X_support, Y_support, disp)
Y2_masses = generate_Y_masses(X2_masses, X_support, Y_support, disp)

# discrete inverse cumulative functions
def create_discrete_inv_cdf(masses, support):
    """create discrete inverse cumulative function"""
    cumsum = np.cumsum(masses)
    
    def F_inv(u):
        # find the lowest n in support such that u < cumulative sum up to n
        if hasattr(u, '__len__'):
            # vectorized version
            result = np.zeros_like(u)
            for i, u_val in enumerate(u):
                idx = np.searchsorted(cumsum, u_val, side='right')
                idx = min(idx, len(support) - 1)
                result[i] = support[idx]
            return result
        else:
            # scalar version
            idx = np.searchsorted(cumsum, u, side='right')
            idx = min(idx, len(support) - 1)
            return support[idx]
    
    return F_inv

F_inv_X1 = create_discrete_inv_cdf(X1_masses, X_support)
F_inv_X2 = create_discrete_inv_cdf(X2_masses, X_support)
F_inv_Y1 = create_discrete_inv_cdf(Y1_masses, Y_support)
F_inv_Y2 = create_discrete_inv_cdf(Y2_masses, Y_support)


# generate synthetic sample
# u maps to x, v maps to y through the inverse cumulatives
def random_sample(n_points, monotone):
    u = np.random.random((n_points, 1 if monotone else 2))
    v = np.random.random((n_points, 2))
    
    if monotone:
        x = np.array([F_inv_X1(u[:,0]), F_inv_X2(u[:,0])]).T
    else:
        x = np.array([F_inv_X1(u[:,0]), F_inv_X2(u[:,1])]).T
    y = np.array([F_inv_Y1(v[:,0]), F_inv_Y2(v[:,1])]).T
                  
    return u, v, x, y
    

# cost function
def cost_f(x, y):
    return np.exp(y[:,0] + y[:,1])


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
            u, v, x, y = random_sample(n_points, monotone = monotone)
            c = cost_f(x, y)
            ws = vmot.generate_working_sample_T2(u, v, x, y, c)
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
            u, v, x, y = random_sample(n_points, monotone = monotone)
            c = cost_f(x, y)
            ws = vmot.generate_working_sample_T2(u, v, x, y, c)
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