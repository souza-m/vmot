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
import itertools
from scipy.stats import norm
import pickle

import vmot
import option_implied_inverse_cdf as empirical
import torch   # used only for to-cuda when models area loaded

# cost function to be minimized
A = 0
B = 1
def cost_f(x, y):
    # cost = A.x1.x2 + B.y1.y2
    return A * x[:,0] * x[:,1] + B * y[:,0] * y[:,1]

# (-cost), to be maximized
def minus_cost_f(x, y):
    return -cost_f(x, y)

# utils - random (u,v) numbers from hypercube
def random_uvset(n_points, d):
    uniform_sample = np.random.random((n_points, 2*d))
    return uniform_sample

def random_uvset_mono(n_points, d):
    uniform_sample = np.random.random((n_points, d+1))
    return uniform_sample

# utils - grid (u,v) numbers from hypercube
def grid_uvset(n_points, d):
    n_grid = np.array(list(itertools.product(*[list(range(n)) for i in range(2 * d)])))
    uv_set = (2 * n_grid + 1) / (2 * n)   # points in the d-hypercube
    return uv_set

def grid_uvset_mono(n_points, d, compensate=False):
    if compensate:
        n_grid = np.array(list(itertools.product(*([list(range(n**2))] + [list(range(n)) for i in range(d)]))))   # compensate for smaller size with greater number of x points
        uv_set = (2 * n_grid + 1) / (2 * n)   # points in the d-hypercube
        uv_set[:,0] = uv_set[:,0] / n
    else:
        n_grid = np.array(list(itertools.product(*[list(range(n)) for i in range(d+1)])))
        uv_set = (2 * n_grid + 1) / (2 * n)   # points in the d-hypercube
    return uv_set

# utils - file dump
_dir = './model_dump/'
_file_prefix = 'results_'
_file_suffix = '.pickle'

def dump_results(results, label='test'):
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
d = 2
n = 40   # marginal sample/grid size
n_points = n ** (2 * d)
print(f'd: {d}')
print(f'sample size: {n_points}')
opt_parameters = { 'penalization'    : 'L2',
                   'beta_multiplier' : 1,
                   'gamma'           : 100,
                   'batch_size'      : 10000,   # no special formula for this
                   'macro_epochs'    : 10,
                   'micro_epochs'    : 10      }

# sets of (u,v) points
uvset1 = random_uvset(n_points, d)
uvset2 = random_uvset_mono(n_points, d)
uvset3 = grid_uvset(n, d)
uvset4 = grid_uvset_mono(n, d)
print('sample shapes')
print('independent random  ', uvset1.shape)
print('monotone random     ', uvset2.shape)
print('independent grid    ', uvset3.shape)
print('monotone grid       ', uvset4.shape)


# example 1 - normal marginals

# choose scales
sig1 = 1.0
sig2 = 1.0
rho1 = np.sqrt(2.0)
rho2 = np.sqrt(3.0)
x_normal_scale = [sig1, sig2]
y_normal_scale = [rho1, rho2]

# reference value (see proposition)
lam1 = np.sqrt(rho1 ** 2 - sig1 ** 2)
lam2 = np.sqrt(rho2 ** 2 - sig2 ** 2)
ref_value = (A + B) * sig1 * sig2 + B * lam1 * lam2
print(f'normal marginals exact solution = {ref_value:8.4f}')

# inverse cumulatives
def normal_inv_cum_xi(q, i):
    return norm.ppf(q) * x_normal_scale[i]

def normal_inv_cum_yi(q, i):
    return norm.ppf(q) * y_normal_scale[i]

def normal_inv_cum_x(q):
    z = norm.ppf(q)
    return np.array([z * x_normal_scale[i] for i in range(d)]).T

# working samples
ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, normal_inv_cum_xi, normal_inv_cum_yi, minus_cost_f)
ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, normal_inv_cum_x, normal_inv_cum_yi, minus_cost_f)
ws3, xyset3 = vmot.generate_working_sample_uv(uvset3, normal_inv_cum_xi, normal_inv_cum_yi, minus_cost_f)
ws4, xyset4 = vmot.generate_working_sample_uv_mono(uvset4, normal_inv_cum_x, normal_inv_cum_yi, minus_cost_f)

# train/store/load
model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.mtg_train(ws1, opt_parameters, monotone = False, verbose = 100)
model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.mtg_train(ws2, opt_parameters, monotone = True, verbose = 100)
model3, D_evo3, H_evo3, P_evo3, ds_evo3, hs_evo3 = vmot.mtg_train(ws3, opt_parameters, monotone = False, verbose = 100)
model4, D_evo4, H_evo4, P_evo4, ds_evo4, hs_evo4 = vmot.mtg_train(ws4, opt_parameters, monotone = True, verbose = 100)
dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1,
              model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2,
              model3, D_evo3, H_evo3, P_evo3, ds_evo3, hs_evo3,
              model4, D_evo4, H_evo4, P_evo4, ds_evo4, hs_evo4  ], 'normal')
# model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1, model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2, model3, D_evo3, H_evo3, P_evo3, ds_evo3, hs_evo3, model4, D_evo4, H_evo4, P_evo4, ds_evo4, hs_evo4 = load_results('normal')

# plot
evo1 = -np.array(D_evo1) # random, independent
evo2 = -np.array(D_evo2) # random, monotone
evo3 = -np.array(D_evo3) # grid, independent
evo4 = -np.array(D_evo4) # grid, monotone
convergence_plot([evo2, evo1, evo4, evo3], ['monotone', 'original', 'grid-monotone', 'grid-independent'], ref_value)


# example 2 - empirical

AMZN_inv_cdf = empirical.AMZN_inv_cdf
AAPL_inv_cdf = empirical.AAPL_inv_cdf

def empirical_inv_cum_xi(q, i):
    if i == 0:
        return AMZN_inv_cdf[0](q)
    if i == 1:
        return AAPL_inv_cdf[0](q)

def empirical_inv_cum_yi(q, i):
    if i == 0:
        return AMZN_inv_cdf[1](q)
    if i == 1:
        return AAPL_inv_cdf[1](q)

def empirical_inv_cum_x(q):
    return np.array([empirical_inv_cum_xi(q, i) for i in range(d)]).T

# working samples
ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, empirical_inv_cum_xi, empirical_inv_cum_yi, minus_cost_f)
ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, empirical_inv_cum_x, empirical_inv_cum_yi, minus_cost_f)
ws3, xyset3 = vmot.generate_working_sample_uv(uvset3, empirical_inv_cum_xi, empirical_inv_cum_yi, minus_cost_f)
ws4, xyset4 = vmot.generate_working_sample_uv_mono(uvset4, empirical_inv_cum_x, empirical_inv_cum_yi, minus_cost_f)
sample_mean_cost = -0.5 * (ws1[:,-2].mean() + ws2[:,-2].mean())   # lower reference for the optimal cost

# train/store/load
model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.mtg_train(ws1, opt_parameters, monotone = False, verbose = 100)
model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.mtg_train(ws2, opt_parameters, monotone = True, verbose = 100)
model3, D_evo3, H_evo3, P_evo3, ds_evo3, hs_evo3 = vmot.mtg_train(ws3, opt_parameters, monotone = False, verbose = 100)
model4, D_evo4, H_evo4, P_evo4, ds_evo4, hs_evo4 = vmot.mtg_train(ws4, opt_parameters, monotone = True, verbose = 100)
dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1,
              model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2,
              model3, D_evo3, H_evo3, P_evo3, ds_evo3, hs_evo3,
              model4, D_evo4, H_evo4, P_evo4, ds_evo4, hs_evo4  ], 'empirical')
# model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1, model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2, model3, D_evo3, H_evo3, P_evo3, ds_evo3, hs_evo3, model4, D_evo4, H_evo4, P_evo4, ds_evo4, hs_evo4 = load_results('empirical')

# plot
evo1 = -np.array(D_evo1)
evo2 = -np.array(D_evo2)
convergence_plot([evo2, evo1], ['monotone', 'original'], sample_mean_cost)




# compensate size in grid mono
uvset5 = grid_uvset_mono(n, d, compensate=True)
print('monotone grid       ', uvset5.shape)
ws5, xyset5 = vmot.generate_working_sample_uv_mono(uvset5, normal_inv_cum_x, normal_inv_cum_yi, minus_cost_f)
model5_normal, D_evo5_normal, H_evo5_normal, P_evo5_normal, ds_evo5_normal, hs_evo5_normal = vmot.mtg_train(ws5, opt_parameters, monotone = True, verbose = 100)
ws5, xyset5 = vmot.generate_working_sample_uv_mono(uvset5, empirical_inv_cum_x, empirical_inv_cum_yi, minus_cost_f)
model5_empirical, D_evo5_empirical, H_evo5_empirical, P_evo5_empirical, ds_evo5_empirical, hs_evo5_empirical = vmot.mtg_train(ws5, opt_parameters, monotone = True, verbose = 100)




# test mode - reiterate train recycling the model
# note: load correct working samples above before running

_model1, _D_evo1, _H_evo1, _P_evo1, _ds_evo1, _hs_evo1 = vmot.mtg_train(ws1, opt_parameters, model = model1, monotone = False, verbose = 100)
_model2, _D_evo2, _H_evo2, _P_evo2, _ds_evo2, _hs_evo2 = vmot.mtg_train(ws2, opt_parameters, model = model2, monotone = True, verbose = 100)
_model3, _D_evo3, _H_evo3, _P_evo3, _ds_evo3, _hs_evo3 = vmot.mtg_train(ws3, opt_parameters, model = model1, monotone = False, verbose = 100)
_model4, _D_evo4, _H_evo4, _P_evo4, _ds_evo4, _hs_evo4 = vmot.mtg_train(ws4, opt_parameters, model = model2, monotone = True, verbose = 100)

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

D_evo3  = D_evo3  + _D_evo3
H_evo3  = H_evo3  + _H_evo3
P_evo3  = P_evo3  + _P_evo3
ds_evo3 = ds_evo3 + _ds_evo3
hs_evo3 = hs_evo3 + _hs_evo3
model3 = _model3

D_evo4  = D_evo4  + _D_evo4
H_evo4  = H_evo4  + _H_evo4
P_evo4  = P_evo4  + _P_evo4
ds_evo4 = ds_evo4 + _ds_evo4
hs_evo4 = hs_evo4 + _hs_evo4
model4 = _model4

dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1, model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2], 'test')
# dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1, model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2], 'normal')
# dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1, model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2], 'empirical')
# model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1, model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = load_results('test')
