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
from scipy.stats import norm
import pickle

import vmot
import option_implied_inverse_cdf as empirical


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
def generate_uvset(n_points, d):
    uniform_sample = np.random.random((n_points, 2*d))
    return uniform_sample

def generate_uvset_mono(n_points, d):
    uniform_sample = np.random.random((n_points, d+1))
    return uniform_sample

# utils - file dump
_dir = './model_dump/'
_file_prefix = 'results_'
_file_suffix = '.pickle'

def dump_results(results, label=''):
    _path = _dir + _file_prefix + label + _file_suffix
    with open(_path, 'wb') as file:
        pickle.dump(results, file)
    print('model saved to ' + _path)

def load_results(label=''):
    _path = _dir + _file_prefix + label + _file_suffix
    with open(_path, 'rb') as file:
        results = pickle.load(file)
    print('model loaded from ' + _path)
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
# full_size = n ** (2 * d)
# mono_size = n ** (d + 1)
full_size = 2000000
mono_size = full_size
print(f'd: {d}')
print(f'full size: {full_size}')
print(f'mono size: {mono_size}')


# example 1 - normal marginals

# choose scales
sig1 = 1.0
sig2 = 1.0
rho1 = np.sqrt(2.0)
rho2 = np.sqrt(3.0)
x_normal_scale = [sig1, sig2]
y_normal_scale = [rho1, rho2]

# reference value (Prop. 3.7)
lam1 = np.sqrt(rho1 ** 2 - sig1 ** 2)
lam2 = np.sqrt(rho2 ** 2 - sig2 ** 2)
ref_value = (A + B) * sig1 * sig2 + B * lam1 * lam2
print(f'normal marginals exact solution = {ref_value:8.4f}')

# 1.1. normal, independent
def normal_inv_cum_xi(q, i):
    return norm.ppf(q) * x_normal_scale[i]

def normal_inv_cum_yi(q, i):
    return norm.ppf(q) * y_normal_scale[i]

n_points = full_size
np.random.seed(1)
uvset1 = generate_uvset(n_points, d)
ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, normal_inv_cum_xi, normal_inv_cum_yi, minus_cost_f, uniform_weight = True)
print('independent coupling sample shape ', uvset1.shape)

# 1.2. normal, monotone
def normal_inv_cum_x(q):
    z = norm.ppf(q)
    return np.array([z * x_normal_scale[i] for i in range(d)]).T

n_points = mono_size
np.random.seed(1)
uvset2 = generate_uvset_mono(n_points, d)
ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, normal_inv_cum_x, normal_inv_cum_yi, minus_cost_f, uniform_weight = True)
print('monotone coupling sample shape ', uvset2.shape)

# train and store
opt_parameters = { 'penalization'    : 'L2',
                   'beta_multiplier' : 1,
                   'gamma'           : 100,
                   'batch_size'      : n ** d,   # no special formula for this, using sqrt of working sample size
                   'macro_epochs'    : 2,
                   'micro_epochs'    : 2      }

# model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.mtg_train(ws1, opt_parameters, monotone = False, verbose = 100)
# model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.mtg_train(ws2, opt_parameters, monotone = True, verbose = 100)

# dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1, model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2], 'normal')
model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1, model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = load_results('normal')

# plot
evo1 = -np.array(D_evo1)
evo2 = -np.array(D_evo2)
convergence_plot([evo2, evo1], ['monotone', 'original'], ref_value)

evo1 = -np.array(D_evo1)
evo2 = -np.array(D_evo2)
std1 = -np.array(ds_evo1)
std2 = -np.array(ds_evo2)
convergence_plot_std([evo2, evo1], [std2, std1], ['monotone', 'original'], ref_value)


# example 2 - empirical

AMZN_inv_cdf = empirical.AMZN_inv_cdf
AAPL_inv_cdf = empirical.AAPL_inv_cdf

# 1.1. normal, independent
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

n_points = full_size
np.random.seed(1)
uvset1 = generate_uvset(n_points, d)
ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, empirical_inv_cum_xi, empirical_inv_cum_yi, minus_cost_f, uniform_weight = True)
print(f'empirical full version unconditional mean cost = {ws1[:,-2].mean():8.4f}')


# 1.2. normal, monotone
def empirical_inv_cum_x(q):
    return np.array([empirical_inv_cum_xi(q, i) for i in range(d)]).T

n_points = mono_size
np.random.seed(1)
uvset2 = generate_uvset_mono(n_points, d)
ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, empirical_inv_cum_x, empirical_inv_cum_yi, minus_cost_f, uniform_weight = True)
print(f'empirical mono version unconditional mean cost = {ws2[:,-2].mean():8.4f}')

vmot.plot_sample_2d(xyset1, label='empirical')
vmot.plot_sample_2d(xyset2, label='empirical, monotone')

# train and store
opt_parameters = { 'penalization'    : 'L2',
                   'beta_multiplier' : 1,
                   'gamma'           : 100,
                   'batch_size'      : n ** d,   # no special formula for this, using sqrt of working sample size
                   'macro_epochs'    : 10,
                   'micro_epochs'    : 2      }

model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.mtg_train(ws1, opt_parameters, monotone = False, verbose = 100)
model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.mtg_train(ws2, opt_parameters, monotone = True, verbose = 100)

dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1, model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2], 'empirical')

# plot
unconditional_mean = -0.5 * (ws1[:,-2].mean() + ws2[:,-2].mean())
evo1 = -np.array(D_evo1)
evo2 = -np.array(D_evo2)
convergence_plot([evo2, evo1], ['monotone', 'original'], unconditional_mean)


# test mode - reiterate train recycling the model
opt_parameters = { 'penalization'    : 'L2',
                   'beta_multiplier' : 1,
                   'gamma'           : 100,
                   'batch_size'      : n ** d,   # no special formula for this, using sqrt of working sample size
                   'macro_epochs'    : 10,
                   'micro_epochs'    : 2      }
_model1, _D_evo1, _H_evo1, _P_evo1, _ds_evo1, _hs_evo1 = vmot.mtg_train(ws1, opt_parameters, model = model1, monotone = False, verbose = 100)
_model2, _D_evo2, _H_evo2, _P_evo2, _ds_evo2, _hs_evo2 = vmot.mtg_train(ws2, opt_parameters, model = model2, monotone = True, verbose = 100)
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

# dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1, model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2], 'test')
dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1, model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2], 'normal')
# dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1, model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2], 'empirical')
# model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1, model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = load_results('test')
