# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:35:25 2021
@author: souzam
PyTorch implementation of Eckstein and Kupper 2021 - Computation of Optimal Transport...
"""

# empirical marginals, d = 2
# Example 1:   max cross-product
# Example 2:   min cross-product
#
#   cross-product function (to be maximized):   cost_f(x, y) = y1 * y2
#
# Coupling structure:
#   (1)   independent (4 dimensions))
#   (2)   montone, dimension reduction on x (3 dimensions)
#
# Method: use vmot_core to generate "working sample" objects to be used in the optimization loop.
# The optimization loop performs the dual approximation based on EK21.
# We experiment with dimensionlality reduction and random vs grid sampling methods.


import numpy as np
import matplotlib.pyplot as pl
from cycler import cycler

import vmot
import option_implied_inverse_cdf as empirical


# processing parameters
d = 2
n_points = 1000000
print(f'd: {d}')
print(f'sample size: {n_points}')
opt_parameters = { 'penalization'    : 'L2',
                   'beta_multiplier' : 1,
                   'gamma'           : 100,
                   'batch_size'      : 2000,   # no special formula for this
                   'epochs'          : 10      }

# cost function to be maximized
A = 0
B = 1
def cost_f(x, y):
    # cost = A.x1.x2 + B.y1.y2
    return A * x[:,0] * x[:,1] + B * y[:,0] * y[:,1]

# negative, used to minimize the cost
def minus_cost_f(x, y):
    return -cost_f(x, y)


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


# process
# # example = 1
# for gamma in [200]:
#  opt_parameters['gamma'] = gamma
#  for example in [1, 2]:
#     if example == 1:
#         cost = cost_f
#         label = f'empirical_g{gamma:d}'
#     if example == 2:
#         cost = minus_cost_f
#         label = f'minus_empirical_g{gamma:d}'
#     print(label)
#     I = 10           # number of desired iterations
#     existing_i = 10
#     np.random.seed(1)
#     print(f'gamma = {gamma:d}')
    
#     if existing_i == 0:
#         # train & dump
#         print('\niteration 1 (new model)\n')
#         uvset1  = vmot.random_uvset(n_points, d)
#         uvset2  = vmot.random_uvset_mono(n_points, d)
#         ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, empirical_inv_cum_xi, empirical_inv_cum_yi, cost_f)
#         ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, empirical_inv_cum_x, empirical_inv_cum_yi, cost_f)
#         model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.mtg_train(ws1, opt_parameters, monotone = False, verbose = 10)
#         model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.mtg_train(ws2, opt_parameters, monotone = True, verbose = 10)
#         existing_i = 1
#         print('models generated')
#         vmot.dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1], label + '_1')
#         vmot.dump_results([model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2], label + '_mono_1')
#     else:
#         # load
#         print(f'\nloading model {existing_i}')
#         model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.load_results(label + f'_{existing_i}')
#         model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.load_results(label + f'_mono_{existing_i}')
    
#     # iterate optimization
#     while existing_i < I:
        
#         # new random sample
#         print(f'\niteration {existing_i+1}\n')
#         uvset1 = vmot.random_uvset(n_points, d)
#         uvset2 = vmot.random_uvset_mono(n_points, d)
#         # ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, empirical_inv_cum_xi, empirical_inv_cum_yi, cost_f)
#         # ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, empirical_inv_cum_x, empirical_inv_cum_yi, cost_f)
#         ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, empirical_inv_cum_xi, empirical_inv_cum_yi, cost_f)
#         ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, empirical_inv_cum_x, empirical_inv_cum_yi, cost_f)
        
#         # train
#         _model1, _D_evo1, _H_evo1, _P_evo1, _ds_evo1, _hs_evo1 = vmot.mtg_train(ws1, opt_parameters, model=model1, monotone = False, verbose = 10)
#         _model2, _D_evo2, _H_evo2, _P_evo2, _ds_evo2, _hs_evo2 = vmot.mtg_train(ws2, opt_parameters, model=model2, monotone = True, verbose = 10)
        
#         # concatenate
#         D_evo1  = D_evo1  + _D_evo1
#         H_evo1  = H_evo1  + _H_evo1
#         P_evo1  = P_evo1  + _P_evo1
#         ds_evo1 = ds_evo1 + _ds_evo1
#         hs_evo1 = hs_evo1 + _hs_evo1
#         model1 = _model1
#         D_evo2  = D_evo2  + _D_evo2
#         H_evo2  = H_evo2  + _H_evo2
#         P_evo2  = P_evo2  + _P_evo2
#         ds_evo2 = ds_evo2 + _ds_evo2
#         hs_evo2 = hs_evo2 + _hs_evo2
#         model2 = _model2
#         existing_i = existing_i + 1
#         print('models updated')
        
#         # dump
#         vmot.dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1], label + f'_{existing_i}')
#         vmot.dump_results([model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2], label + f'_mono_{existing_i}')
        
#         # plot
#         # evo1 = np.array(D_evo1) # random, independent
#         # evo2 = np.array(D_evo2) # random, monotone
#         # vmot.convergence_plot([-evo2, -evo1], ['reduced', 'full'], ref_value=-sample_mean_cost)
#         # vmot.convergence_plot([-evo2, -evo1], ['reduced', 'full'], ref_value=-sample_mean_cost)

for example in [1, 2]:
    if example == 1:
        cost = cost_f
        label = 'empirical_vg'
    if example == 2:
        cost = minus_cost_f
        label = 'minus_empirical_vg'
    print(label)
    I = 20           # number of desired iterations
    existing_i = 0
    np.random.seed(1)
    
    if existing_i == 0:
        # train & dump
        print('\niteration 1 (new model)\n')
        uvset1  = vmot.random_uvset(n_points, d)
        uvset2  = vmot.random_uvset_mono(n_points, d)
        ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, empirical_inv_cum_xi, empirical_inv_cum_yi, cost)
        ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, empirical_inv_cum_x, empirical_inv_cum_yi, cost)
        opt_parameters['gamma'] = 100 * (existing_i+1)
        print(f'gamma = {opt_parameters["gamma"]:d}')
        model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.mtg_train(ws1, opt_parameters, monotone = False, verbose = 10)
        model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.mtg_train(ws2, opt_parameters, monotone = True, verbose = 10)
        existing_i = 1
        print('models generated')
        vmot.dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1], label + '_1')
        vmot.dump_results([model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2], label + '_mono_1')
    else:
        # load
        print(f'\nloading model {existing_i}')
        model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.load_results(label + f'_{existing_i}')
        model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.load_results(label + f'_mono_{existing_i}')
    
    # iterate optimization
    while existing_i < I:
        
        # new random sample
        print(f'\niteration {existing_i+1}\n')
        uvset1 = vmot.random_uvset(n_points, d)
        uvset2 = vmot.random_uvset_mono(n_points, d)
        opt_parameters['gamma'] = 100 * (existing_i+1)
        print(f'gamma = {opt_parameters["gamma"]:d}')
        ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, empirical_inv_cum_xi, empirical_inv_cum_yi, cost)
        ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, empirical_inv_cum_x, empirical_inv_cum_yi, cost)
        
        # train
        _model1, _D_evo1, _H_evo1, _P_evo1, _ds_evo1, _hs_evo1 = vmot.mtg_train(ws1, opt_parameters, model=model1, monotone = False, verbose = 10)
        _model2, _D_evo2, _H_evo2, _P_evo2, _ds_evo2, _hs_evo2 = vmot.mtg_train(ws2, opt_parameters, model=model2, monotone = True, verbose = 10)
        
        # concatenate
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
        
        # dump
        vmot.dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1], label + f'_{existing_i}')
        vmot.dump_results([model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2], label + f'_mono_{existing_i}')
        
        # plot
        # evo1 = np.array(D_evo1) # random, independent
        # evo2 = np.array(D_evo2) # random, monotone
        # vmot.convergence_plot([-evo2, -evo1], ['reduced', 'full'], ref_value=-sample_mean_cost)
        # vmot.convergence_plot([-evo2, -evo1], ['reduced', 'full'], ref_value=-sample_mean_cost)
    
    
# --- report mean and std over a collection of samples ---

# store cost and calculate (unconditional) sample mean, if used
cost_series = None
sample_mean_cost = 0.0   # lower reference for the optimal cost

report = True
if report:
    collection_size = 5 #10
    D1_series = []
    D2_series = []
    P1_series = []
    P2_series = []
    for i in range(collection_size):
        print(i)
        uvset1 = vmot.random_uvset(n_points, d)
        uvset2 = vmot.random_uvset_mono(n_points, d)
        # ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, empirical_inv_cum_xi, empirical_inv_cum_yi, cost_f)
        # ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, empirical_inv_cum_x, empirical_inv_cum_yi, cost_f)
        ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, empirical_inv_cum_xi, empirical_inv_cum_yi, minus_cost_f)
        ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, empirical_inv_cum_x, empirical_inv_cum_yi, minus_cost_f)
        if cost_series is None: 
            cost_series = np.hstack([ws1[:,-2], ws2[:,-2]])
        else:
            cost_series = np.hstack([cost_series, ws1[:,-2], ws2[:,-2]])
        D1, P1, __ = vmot.mtg_dual_value(model1, ws1, opt_parameters, normalize_pi = False)
        D2, P2, __ = vmot.mtg_dual_value(model2, ws2, opt_parameters, normalize_pi = False)
        D1_series.append(D1)
        D2_series.append(D2)
        P1_series.append(P1)
        P2_series.append(P2)
    sample_mean_cost = cost_series.mean()   # central reference for the upper and lower bounds
    print(f'sample mean cost = {sample_mean_cost:7.4f}')
    print('dual value')
    print(f'full:     mean = {np.mean(D1_series):8.4f};   std = {np.std(D1_series):8.4f}')
    print(f'reduced:  mean = {np.mean(D2_series):8.4f};   std = {np.std(D2_series):8.4f}')
    print('penalty')
    print(f'full:     mean = {np.mean(P1_series):8.4f};   std = {np.std(P1_series):8.4f}')
    print(f'reduced:  mean = {np.mean(P2_series):8.4f};   std = {np.std(P2_series):8.4f}')


# --- low-information bounds ---
# for the sake of comparison, calculate p+ and p- when only the t=2 distribution is known

def single_period_cost_f(y):
    # cost = A.x1.x2 + B.y1.y2
    return y[:,0] * y[:,1]

# monospaced diagonal
diag_n  = int(1e7)
diag_grid = np.array(range(diag_n))
uv_set = (2 * diag_grid + 1) / (2 * diag_n)   # points in the d-hypercube

# mean costs
positive_diag_y = np.vstack([empirical_inv_cum_yi(uv_set, 0), empirical_inv_cum_yi(uv_set, 1)]).T
positive_cost = single_period_cost_f(positive_diag_y)
mean_positive_cost = positive_cost.mean()
negative_diag_y = np.vstack([empirical_inv_cum_yi(uv_set[::-1], 0), empirical_inv_cum_yi(uv_set, 1)]).T
negative_cost = single_period_cost_f(negative_diag_y)
mean_negative_cost = negative_cost.mean()

# check cost array
pl.figure()
pl.plot(positive_cost)
pl.plot(negative_cost)
pl.legend(['positive', 'negative'])





# load
uvset1  = vmot.random_uvset(n_points, d)
uvset2  = vmot.random_uvset_mono(n_points, d)

label, label_mono = 'empirical_vg_20', 'empirical_vg_mono_20'
# ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, empirical_inv_cum_xi, empirical_inv_cum_yi, cost_f)
# ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, empirical_inv_cum_x, empirical_inv_cum_yi, cost_f)
# model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.load_results(label)
# model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.load_results('minus_' + label)    

suffixes = ['vg', 'g5000']
suffix = suffixes[1]
existing_i = 10
# --- plot all bounds ---
# existing_i=10
a_, D_evo1_plus, _, __, ___, ____ = vmot.load_results(label)
b_, D_evo2_plus, _, __, ___, ____ = vmot.load_results(label_mono)
existing_i=10
a_, D_evo1_minus, _, __, ___, ____ = vmot.load_results('minus_' + label)
b_, D_evo2_minus, _, __, ___, ____ = vmot.load_results('minus_' + label_mono)

evo_a1 = np.array(D_evo1_plus) # random, independent
evo_a2 = np.array(D_evo2_plus) # random, monotone
evo_b1 = -np.array(D_evo1_minus) # random, independent
evo_b2 = -np.array(D_evo2_minus) # random, monotone
labels = ['full', 'reduced']

# choose style
# pl.style.use('classic')
# pl.style.use('c')
# pl.style.use('ggplot')
# pl.style.use('seaborn-white')
# pl.rcParams['image.cmap'] = 'jet'   # does not work
# print(pl.rcParams['axes.prop_cycle'].by_key()['color'])
# pl.gca().set_prop_cycle(None)   # default color cycler

# chosen color cycler (copied from 'bmh' style)
cc = cycler('color', ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00', '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2'])

pl.figure(figsize = [5,5])
pl.gca().set_prop_cycle(cc)
for v in [evo_a1, evo_a2]:
    pl.plot(range(1, len(v)+1), v)
pl.gca().set_prop_cycle(cc)
for v in [evo_b1, evo_b2]:
    pl.plot(range(1, len(v)+1), v)
pl.legend(labels, loc='lower right')

pl.axhline(mean_positive_cost, linestyle='-', color='grey')
# pl.axhline(sample_mean_cost, linestyle=':', color='black')
pl.axhline(mean_negative_cost, linestyle='-', color='grey')

# pl.title(title)
shift = .035 * (pl.gca().get_ylim()[1] - pl.gca().get_ylim()[0])
# pl.annotate('sample mean', (len(v)*1/4, sample_mean_cost-shift), color='black')   # trial and error to find a good position
pl.annotate(r'$p^+$', (len(v)*3/4, D_evo1_plus[-1]+shift))   # trial and error to find a good position
pl.annotate(r'$p^-$', (len(v)*3/4, -D_evo1_minus[-1]-2*shift))   # trial and error to find a good position
pl.tight_layout()
pl.show()


# closing prices Dec 16th
p1 = 134.51  # AAPL  https://finance.yahoo.com/quote/AAPL/history?period1=1669852800&period2=1672444800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true
p2 =  87.86  # AMZN  https://finance.yahoo.com/quote/AMZN/history?period1=1669852800&period2=1672444800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true







