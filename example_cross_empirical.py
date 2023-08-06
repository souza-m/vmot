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
                   'gamma'           : 1000,
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

p0_AAPL = 134.51  # https://finance.yahoo.com/quote/AAPL/history?period1=1669852800&period2=1672444800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true
p0_AMZN =  87.86  # https://finance.yahoo.com/quote/AMZN/history?period1=1669852800&period2=1672444800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true

def empirical_inv_cum_xi(q, i):
    if i == 0:
        return (AMZN_inv_cdf[0](q) / p0_AMZN) - 1.0
    if i == 1:
        return (AAPL_inv_cdf[0](q) / p0_AAPL) - 1.0

def empirical_inv_cum_yi(q, i):
    if i == 0:
        return (AMZN_inv_cdf[1](q) / p0_AMZN) - 1.0
    if i == 1:
        return (AAPL_inv_cdf[1](q) / p0_AAPL) - 1.0

def empirical_inv_cum_x(q):
    return np.array([empirical_inv_cum_xi(q, i) for i in range(d)]).T


# process
# example = 1
# for gamma in [2000, 5000, 10000, 1000, 500]:
for gamma in [100, 1000]:
 for example in [1, 2]:
    if example == 1:
        cost = cost_f
        label = f'plus_empirical_ret_gamma{gamma:d}i'
    if example == 2:
        cost = minus_cost_f
        label = f'minus_empirical_ret_gamma{gamma:d}i'
    print(label)
    I = 30           # number of desired iterations
    existing_i = 0
    opt_parameters['gamma'] = gamma * (existing_i+1)
    print(opt_parameters['gamma'])
    np.random.seed(1)
    # print(f'gamma = {gamma:d}')
    
    if existing_i == 0:
        # train & dump
        print('\niteration 1 (new model)\n')
        uvset1  = vmot.random_uvset(n_points, d)
        uvset2  = vmot.random_uvset_mono(n_points, d)
        ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, empirical_inv_cum_xi, empirical_inv_cum_yi, cost)
        ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, empirical_inv_cum_x, empirical_inv_cum_yi, cost)
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
        opt_parameters['gamma'] = gamma * (existing_i+1)
        print(opt_parameters['gamma'])

        # new random sample
        print(f'\niteration {existing_i+1}\n')
        uvset1 = vmot.random_uvset(n_points, d)
        uvset2 = vmot.random_uvset_mono(n_points, d)
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
        
for gamma in [10000, 100000, 10, 1]:
 for example in [1, 2]:
    if example == 1:
        cost = cost_f
        label = f'plus_empirical_ret_gamma{gamma:d}'
    if example == 2:
        cost = minus_cost_f
        label = f'minus_empirical_ret_gamma{gamma:d}'
    print(label)
    I = 20           # number of desired iterations
    existing_i = 0
    opt_parameters['gamma'] = gamma
    print(gamma)
    np.random.seed(1)
    # print(f'gamma = {gamma:d}')
    
    if existing_i == 0:
        # train & dump
        print('\niteration 1 (new model)\n')
        uvset1  = vmot.random_uvset(n_points, d)
        uvset2  = vmot.random_uvset_mono(n_points, d)
        ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, empirical_inv_cum_xi, empirical_inv_cum_yi, cost)
        ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, empirical_inv_cum_x, empirical_inv_cum_yi, cost)
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
        print(gamma)

        # new random sample
        print(f'\niteration {existing_i+1}\n')
        uvset1 = vmot.random_uvset(n_points, d)
        uvset2 = vmot.random_uvset_mono(n_points, d)
        # ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, empirical_inv_cum_xi, empirical_inv_cum_yi, cost_f)
        # ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, empirical_inv_cum_x, empirical_inv_cum_yi, cost_f)
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


# --- low-information bounds ---
# for the sake of comparison, calculate p+ and p- when only the t=2 distribution is known

def single_period_cost_f(y):
    # cost = A.x1.x2 + B.y1.y2
    return y[:,0] * y[:,1]

# monospaced diagonal
diag_n  = int(1e7)
diag_grid = np.array(range(diag_n))
uv_set = (2 * diag_grid + 1) / (2 * diag_n)

# mean costs
positive_diag_y = np.vstack([empirical_inv_cum_yi(uv_set, 0), empirical_inv_cum_yi(uv_set, 1)]).T
positive_cost = single_period_cost_f(positive_diag_y)
mean_positive_cost = positive_cost.mean()
negative_diag_y = np.vstack([empirical_inv_cum_yi(uv_set[::-1], 0), empirical_inv_cum_yi(uv_set, 1)]).T
negative_cost = single_period_cost_f(negative_diag_y)
mean_negative_cost = negative_cost.mean()
print(f'positive diag {mean_positive_cost:10.4f}\nnegative diag {mean_negative_cost:10.4f}   ')
# check cost array
pl.figure()
pl.plot(positive_cost)
pl.plot(negative_cost)
pl.legend(['positive', 'negative'])


# load
# uvset1  = vmot.random_uvset(n_points, d)
# uvset2  = vmot.random_uvset_mono(n_points, d)

label, label_mono = 'plus_empirical_ret_gamma100000_20', 'plus_empirical_ret_gamma100000_mono_20'
minus_label, minus_label_mono = 'minus_empirical_ret_gamma100000_20', 'minus_empirical_ret_gamma100000_mono_20'

model1_plus,  D_evo1_plus, _, __, ___, ____ = vmot.load_results(label)
model2_plus,  D_evo2_plus, _, __, ___, ____ = vmot.load_results(label_mono)
model1_minus, D_evo1_minus, _, __, ___, ____ = vmot.load_results(minus_label)
model2_minus, D_evo2_minus, _, __, ___, ____ = vmot.load_results(minus_label_mono)


# report mean and std over a collection of samples

# store cost and calculate (unconditional) sample mean, if used
sample_mean_cost = 0.0   # lower reference for the optimal cost
report = True
if report:
    report_n_points = 1000000
    uvset1 = vmot.random_uvset(report_n_points, d)
    uvset2 = vmot.random_uvset_mono(report_n_points, d)
    ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, empirical_inv_cum_xi, empirical_inv_cum_yi, cost_f)
    ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, empirical_inv_cum_x, empirical_inv_cum_yi, cost_f)
    D1, P1, __ = vmot.mtg_dual_value(model1_plus, ws1, opt_parameters, normalize_pi = False)
    D2, P2, __ = vmot.mtg_dual_value(model2_plus, ws2, opt_parameters, normalize_pi = False)
    cost_series = np.hstack([ws1[:,-2], ws2[:,-2]])
    sample_mean_cost = cost_series.mean()   # central reference for the upper and lower bounds
    print(f'sample mean cost = {sample_mean_cost:7.4f}')
    print('dual value')
    print(f'full:     mean = {np.mean(D1):8.4f};   std = {np.std(D1):8.4f}')
    print(f'reduced:  mean = {np.mean(D2):8.4f};   std = {np.std(D2):8.4f}')
    print('penalty')
    print(f'full:     mean = {np.mean(P1):8.4f};   std = {np.std(P1):8.4f}')
    print(f'reduced:  mean = {np.mean(P2):8.4f};   std = {np.std(P2):8.4f}')


# convergence graph (evolution of the dual value)

evo_a1 = np.array(D_evo1_plus) # random, independent
evo_a2 = np.array(D_evo2_plus) # random, monotone
evo_b1 = -np.array(D_evo1_minus) # random, independent
evo_b2 = -np.array(D_evo2_minus) # random, monotone

# chosen color cycler (copied from 'bmh' style)
cc = cycler('color', ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00', '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2'])

pl.figure(figsize = [5,5])
pl.gca().set_prop_cycle(cc)
for v in [evo_a1, evo_a2]:
    pl.plot(range(1, len(v)+1), v)
pl.gca().set_prop_cycle(cc)
for v in [evo_b1, evo_b2]:
    pl.plot(range(1, len(v)+1), v)
pl.legend(['full', 'reduced'], loc='lower right')

pl.axhline(mean_positive_cost, linestyle='-', color='grey')
pl.axhline(sample_mean_cost, linestyle=':', color='black')
pl.axhline(mean_negative_cost, linestyle='-', color='grey')

# pl.title(title)
shift = .035 * (pl.gca().get_ylim()[1] - pl.gca().get_ylim()[0])
# pl.annotate('sample mean', (len(v)*1/4, sample_mean_cost-shift), color='black')   # trial and error to find a good position
pl.annotate(r'$c^+$', (len(v)*3/4, D_evo1_plus[-1]+shift))   # trial and error to find a good position
pl.annotate(r'$c^-$', (len(v)*3/4, -D_evo1_minus[-1]-2*shift))   # trial and error to find a good position
pl.tight_layout()
pl.show()


