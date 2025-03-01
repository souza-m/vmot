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


# example with cross-product cost and uniform marginals, d = 2, T = 3
# over all possible joint distributions of (X,Y)


# two marginals 1, 2
# three periods X, Y, Z
d = 2
T = 2

# inverse cumulative of U[0.5 , 1.0]
def F_inv_X1(u):
    return 0.5 + 0.5 * u

# inverse cumulative of U[0.5 , 1.0]
def F_inv_X2(u):
    return 0.5 + 0.5 * u

# inverse cumulative of U[0.5 , 1.0]
def F_inv_Y1(u):
    return 0.5 + 0.5 * u

# inverse cumulative of U[0.0 , 1.5]
def F_inv_Y2(u):
    return 1.5 * u


# generate synthetic sample
# u maps to x, v maps to y, w maps to z through the inverse cumulatives
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
            vmot.dump_results([model, opt, D_series], f'uniform_d{d}_T{T}_{existing_i}_' + mono_label)
            
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
            model, opt, D_series = vmot.load_results(f'uniform_d{d}_T{T}_{existing_i}_' + mono_label)
            
            # train
            _D_series, _H_series = vmot.mtg_train(ws, model, opt, d, T, monotone = monotone, opt_parameters=opt_parameters, verbose = 1)
            existing_i += 1
            print('models updated')
            
            # sequential storage of the variables' evolution
            D_series = D_series + _D_series
            H_series = H_series + _H_series
            vmot.dump_results([model, opt, D_series], f'uniform_d{d}_T{T}_{existing_i}_' + mono_label)
    
        t1 = time.time() # timer
        timers.append(t1)
        print('duration = ' + str(dt.timedelta(seconds=round(t1 - t0))))
