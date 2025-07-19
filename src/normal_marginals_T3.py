# -*- coding: utf-8 -*-
"""
Created on Jul 18 2025
@author: souzam
PyTorch implementation of Eckstein and Kupper 2019 - Computation of Optimal Transport...
Example with cross-product cost and normal marginals, d = 2, T = 3
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import vmot_dual as vmot
import matplotlib.pyplot as pl
import datetime as dt, time

# two marginals, three periods
d = 2
T = 3

# Parameters for the normal distributions
mean_X = [0, 0]
std_X = [1, 2]
mean_Y = [0, 0]
std_Y = [1.5, 2.5]

def F_inv_X1(u):
    return norm.ppf(u) * std_X[0] + mean_X[0]

def F_inv_X2(u):
    return norm.ppf(u) * std_X[1] + mean_X[1]

def F_inv_Y1(u):
    return norm.ppf(u) * std_Y[0] + mean_Y[0]

def F_inv_Y2(u):
    return norm.ppf(u) * std_Y[1] + mean_Y[1]

def F_inv_Z1(u):
    return norm.ppf(u) * std_X[0] + mean_X[0]

def F_inv_Z2(u):
    return norm.ppf(u) * std_Y[0] + mean_Y[0]

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

def cost_f(x, y, z):
    return np.exp(z[:,0] + z[:,1])

opt_parameters = { 'gamma' : 10, 'epochs' : 1, 'batch_size' : 1000 }

if __name__ == "__main__":
    I = 100
    n_points = 1000000
    timers = []
    for monotone in [True, False]:
        existing_i = 0
        mono_label = 'mono' if monotone else 'full'
        print(mono_label)
        t0 = time.time()
        timers.append(t0)
        if existing_i == 0:
            print('\niteration 1 (new model)\n')
            u, v, w, x, y, z = random_sample(n_points, monotone = monotone)
            c = cost_f(x, y, z)
            ws = vmot.generate_working_sample_T3(u, v, w, x, y, z, c)
            print('sample generated, shape ', ws.shape)
            model, opt = vmot.generate_model(d, T, monotone = monotone)
            print('model generated')
            D_series, H_series = vmot.mtg_train(ws, model, opt, d, T, monotone = monotone, opt_parameters=opt_parameters, verbose = 1)
            existing_i = 1
            vmot.dump_results([model, opt, D_series], f'normal_d{d}_T{T}_{existing_i}_' + mono_label)
        while existing_i < I:
            print(f'\niteration {existing_i+1}\n')
            u, v, w, x, y, z = random_sample(n_points, monotone = monotone)
            c = cost_f(x, y, z)
            ws = vmot.generate_working_sample_T3(u, v, w, x, y, z, c)
            print('sample generated, shape ', ws.shape)
            print(f'\nloading model {existing_i}')
            model, opt, D_series = vmot.load_results(f'normal_d{d}_T{T}_{existing_i}_' + mono_label)
            _D_series, _H_series = vmot.mtg_train(ws, model, opt, d, T, monotone = monotone, opt_parameters=opt_parameters, verbose = 1)
            existing_i += 1
            print('models updated')
            D_series = D_series + _D_series
            H_series = H_series + _H_series
            vmot.dump_results([model, opt, D_series], f'normal_d{d}_T{T}_{existing_i}_' + mono_label)
        t1 = time.time()
        timers.append(t1)
        print('duration = ' + str(dt.timedelta(seconds=round(t1 - t0))))