# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:35:25 2021
@author: souzam
PyTorch implementation of Eckstein and Kupper 2019 - Computation of Optimal Transport...
"""

# Example 1: solve cross-product cost with normal marginals, d = 2
#
#   Cost function:   cost_f(x, y) = y1 * y2
#
#   Primal:          max cost
#            equiv.  min C := minus_cost
#
#   Dual:            max D := sum{phi} + sum{psi} + sum{h.L}
#                    st  D <= minus_cost
#
#   Penalized dual:  min (-D) + b(D - minus_cost)
#
# Coupling structure:
#   (1)   independent (4 dimensions))
#   (2)   montone, dimension reduction on x (3 dimensions)
#
# Method: use vmot_core to generate "working sample" objects to be used in the optimization loop.
# The optimization loop performs the dual approximation based on E&K21.
# We experiment with dimensionlality reduction and random vs grid sampling methods.


import numpy as np
import matplotlib.pyplot as pl
# import itertools
from scipy.stats import norm
import pickle

import vmot
import torch   # used only for to-cuda when models area loaded

# utils - random (u,v) numbers from hypercube
def random_uvset(n_points, d):
    uniform_sample = np.random.random((n_points, 2*d))
    return uniform_sample

def random_uvset_mono(n_points, d):
    uniform_sample = np.random.random((n_points, d+1))
    return uniform_sample

# utils - grid (u,v) numbers from hypercube
# def grid_uvset(n, d):
#     n_grid = np.array(list(itertools.product(*[list(range(n)) for i in range(2 * d)])))
#     uv_set = (2 * n_grid + 1) / (2 * n)   # points in the d-hypercube
#     return uv_set

# def grid_uvset_mono(n, d):
#     n_grid = np.array(list(itertools.product(*[list(range(n)) for i in range(d+1)])))
#     uv_set = (2 * n_grid + 1) / (2 * n)   # points in the d-hypercube
#     return uv_set

# utils - file dump
_dir = './model_dump/'
_file_prefix = 'results_'
_file_suffix = '.pickle'

def dump_results(results, label='test'):
    # move to cpu before dumping
    cpu_device = torch.device('cpu')
    for i in range(len(results)):
        if isinstance(results[i], torch.nn.modules.container.ModuleList):
            print(i, type(results[i]))
            results[i] = results[i].to(cpu_device)
    
    # dump
    _path = _dir + _file_prefix + label + _file_suffix
    with open(_path, 'wb') as file:
        pickle.dump(results, file)
    print('model saved to ' + _path)

def load_results(label=''):
    _path = _dir + _file_prefix + label + _file_suffix
    with open(_path, 'rb') as file:
        results = pickle.load(file)
    print('model loaded from ' + _path)
    for i in range(len(results)):
        if isinstance(results[i], torch.nn.modules.container.ModuleList):
            print(i, type(results[i]))
            results[i] = results[i].to(vmot.device)
    return results

# utils - convergence plots
def convergence_plot(value_series_list, labels, ref_value = None):
    pl.figure(figsize = [12,12])   # plot in two iterations to have a clean legend
    for v in value_series_list:
        pl.plot(v)
    pl.legend(labels)
    if not ref_value is None:
        pl.axhline(ref_value, linestyle=':', color='black')
    pl.show()

def convergence_plot_std(value_series_list, std_series_list, labels, ref_value = None):
    pl.figure(figsize = [12,12])   # plot in two iterations to have a clean legend
    for v, std in zip(value_series_list, std_series_list):
        pl.plot(v)
    pl.legend(labels)
    for v, std in zip(value_series_list, std_series_list):
        pl.fill_between(range(len(v)), v + std, v - std, alpha = .5, facecolor = 'grey')
    if not ref_value is None:
        pl.axhline(ref_value, linestyle=':', color='black')
    pl.show()


# processing parameters
d = 10
n_points = 2000000
print(f'd: {d}')
print(f'sample size: {n_points}')
opt_parameters = { 'penalization'    : 'L2',
                   'beta_multiplier' : 1,
                   'gamma'           : 100,
                   'batch_size'      : 200,   # no special formula for this
                   'epochs'          : 100      }

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

# (-cost), to be maximized
# def minus_cost_f(x, y):
#     return -cost_f(x, y)

# sets of (u,v) points
uvset1 = random_uvset(n_points, d)
uvset2 = random_uvset_mono(n_points, d)
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

# working samples
ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, normal_inv_cum_xi, normal_inv_cum_yi, cost_f)
ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, normal_inv_cum_x, normal_inv_cum_yi, cost_f)
ws1.shape
ws2.shape

# train/store/load
model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.mtg_train(ws1, opt_parameters, monotone = False, verbose = 100)
model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.mtg_train(ws2, opt_parameters, monotone = True, verbose = 100)
# dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1,
#               model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2  ], 'normal_10')
model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1, model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = load_results('normal_10')

# plot
evo1 = -np.array(D_evo1) # random, independent
evo2 = -np.array(D_evo2) # random, monotone
convergence_plot([evo2, evo1], ['monotone', 'independent'], ref_value)




# tests

_model1, _D_evo1, _H_evo1, _P_evo1, _ds_evo1, _hs_evo1 = vmot.mtg_train(ws1, opt_parameters, monotone = False, verbose = 100)
_model2, _D_evo2, _H_evo2, _P_evo2, _ds_evo2, _hs_evo2 = vmot.mtg_train(ws2, opt_parameters, monotone = True, verbose = 100)
