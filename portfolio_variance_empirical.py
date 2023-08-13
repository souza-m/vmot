# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:35:25 2021
@author: souzam
PyTorch implementation of Eckstein and Kupper 2021 - Computation of Optimal Transport...
"""

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
B = np.array([[.25, .25], [.25, .25]])
def cost_f_(y):
    cost = 0.0
    for i in range(d):
        for j in range(d):
            cost = cost + B[i,j] * y[:,i] * y[:,j]
    return cost

def cost_f(x, y):
    return cost_f_(y)

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

for gamma in [100000, 10000, 1000000]:
 for example in [1, 2]:
    if example == 1:
        cost = cost_f
        label = f'portfolio_empirical_gamma{gamma:d}'
    if example == 2:
        cost = minus_cost_f
        label = f'_portfolio_empirical_gamma{gamma:d}'
    print(label)
    I = 20           # number of desired iterations
    existing_i = 0
    # if gamma <= 10000:
    #     existing_i = 20
    if gamma == 100000:
        existing_i = 20
    if gamma == 10000 and example == 1:
        existing_i = 10
    if gamma == 10000 and example == 2:
        existing_i = 9
    # if gamma == 100000 and example == 2:
    #     existing_i = 15
    opt_parameters['gamma'] = gamma
    print(gamma)
    np.random.seed(0)
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


# MOT bounds

# monospaced diagonal
diag_n  = int(1e7)
diag_grid = np.array(range(diag_n))
uv_set = (2 * diag_grid + 1) / (2 * diag_n)

# mean costs
positive_diag_y = np.vstack([empirical_inv_cum_yi(uv_set, 0), empirical_inv_cum_yi(uv_set, 1)]).T
positive_cost = cost_f_(positive_diag_y)
mean_positive_cost = positive_cost.mean()
negative_diag_y = np.vstack([empirical_inv_cum_yi(uv_set[::-1], 0), empirical_inv_cum_yi(uv_set, 1)]).T
negative_cost = cost_f_(negative_diag_y)
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

plus_label,  plus_label_mono  = 'portfolio_empirical_gamma100000_20', 'portfolio_empirical_gamma100000_mono_20'
minus_label, minus_label_mono = '_portfolio_empirical_gamma100000_20', '_portfolio_empirical_gamma100000_mono_20'

model1_plus,  D_evo1_plus, _, __, ___, ____ = vmot.load_results(plus_label)
model2_plus,  D_evo2_plus, _, __, ___, ____ = vmot.load_results(plus_label_mono)
model1_minus, D_evo1_minus, _, __, ___, ____ = vmot.load_results(minus_label)
model2_minus, D_evo2_minus, _, __, ___, ____ = vmot.load_results(minus_label_mono)


# report mean and std over a collection of samples

report = True
# sample_mean_cost = None   # lower reference for the optimal cost

if report:
    n_points_report = int(1e6)
    collection_size = 10
    cost_series = None
    D1_series = []
    D2_series = []
    P1_series = []
    P2_series = []
    np.random.seed(0)
    for i in range(collection_size):
        uvset1 = vmot.random_uvset(n_points_report, d)
        uvset2 = vmot.random_uvset_mono(n_points_report, d)
        ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, empirical_inv_cum_xi, empirical_inv_cum_yi, cost_f)
        ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, empirical_inv_cum_x, empirical_inv_cum_yi, cost_f)
        # ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, empirical_inv_cum_xi, empirical_inv_cum_yi, minus_cost_f)
        # ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, empirical_inv_cum_x, empirical_inv_cum_yi, minus_cost_f)
        if cost_series is None: 
            cost_series = np.hstack([ws1[:,-2], ws2[:,-2]])
        else:
            cost_series = np.hstack([cost_series, ws1[:,-2], ws2[:,-2]])
        D1, P1, __ = vmot.mtg_dual_value(model1_plus, ws1, opt_parameters, normalize_pi = False)
        D2, P2, __ = vmot.mtg_dual_value(model2_plus, ws2, opt_parameters, normalize_pi = False)
        # D1, P1, __ = vmot.mtg_dual_value(model1_minus, ws1, opt_parameters, normalize_pi = False)
        # D2, P2, __ = vmot.mtg_dual_value(model2_minus, ws2, opt_parameters, normalize_pi = False)
        D1_series.append(D1)
        D2_series.append(D2)
        P1_series.append(P1)
        P2_series.append(P2)
        cost_series = np.hstack([ws1[:,-2], ws2[:,-2]])
    print('dual value (plus)')
    print(f'full:     mean = {np.mean(D1_series):8.4f};   std = {np.std(D1_series):8.4f}')
    print(f'reduced:  mean = {np.mean(D2_series):8.4f};   std = {np.std(D2_series):8.4f}')
    # print('penalty')
    # print(f'full:     mean = {np.mean(P1_series):8.4f};   std = {np.std(P1_series):8.4f}')
    # print(f'reduced:  mean = {np.mean(P2_series):8.4f};   std = {np.std(P2_series):8.4f}')
    
    # only when running "plus"
    sample_mean_cost = cost_series.mean()   # central reference for the upper and lower bounds

    n_points_report = int(1e6)
    collection_size = 10
    cost_series = None
    D1_series = []
    D2_series = []
    P1_series = []
    P2_series = []
    for i in range(collection_size):
        uvset1 = vmot.random_uvset(n_points_report, d)
        uvset2 = vmot.random_uvset_mono(n_points_report, d)
        # ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, empirical_inv_cum_xi, empirical_inv_cum_yi, cost_f)
        # ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, empirical_inv_cum_x, empirical_inv_cum_yi, cost_f)
        ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, empirical_inv_cum_xi, empirical_inv_cum_yi, minus_cost_f)
        ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, empirical_inv_cum_x, empirical_inv_cum_yi, minus_cost_f)
        if cost_series is None: 
            cost_series = np.hstack([ws1[:,-2], ws2[:,-2]])
        else:
            cost_series = np.hstack([cost_series, ws1[:,-2], ws2[:,-2]])
        # D1, P1, __ = vmot.mtg_dual_value(model1_plus, ws1, opt_parameters, normalize_pi = False)
        # D2, P2, __ = vmot.mtg_dual_value(model2_plus, ws2, opt_parameters, normalize_pi = False)
        D1, P1, __ = vmot.mtg_dual_value(model1_minus, ws1, opt_parameters, normalize_pi = False)
        D2, P2, __ = vmot.mtg_dual_value(model2_minus, ws2, opt_parameters, normalize_pi = False)
        D1_series.append(D1)
        D2_series.append(D2)
        P1_series.append(P1)
        P2_series.append(P2)
        cost_series = np.hstack([ws1[:,-2], ws2[:,-2]])
    print('dual value (minus)')
    print(f'full:     mean = {np.mean(D1_series):8.4f};   std = {np.std(D1_series):8.4f}')
    print(f'reduced:  mean = {np.mean(D2_series):8.4f};   std = {np.std(D2_series):8.4f}')
    # print('penalty')
    # print(f'full:     mean = {np.mean(P1_series):8.4f};   std = {np.std(P1_series):8.4f}')
    # print(f'reduced:  mean = {np.mean(P2_series):8.4f};   std = {np.std(P2_series):8.4f}')

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
pl.axhline(mean_negative_cost, linestyle='-', color='grey')
if not sample_mean_cost is None:
    pl.axhline(sample_mean_cost, linestyle=':', color='black')

# pl.title(title)
# shift = .035 * (pl.gca().get_ylim()[1] - pl.gca().get_ylim()[0])
# pl.annotate('sample mean', (len(v)*1/4, sample_mean_cost-shift), color='black')   # trial and error to find a good position
# pl.annotate(r'$c^+$', (len(v)*3/4, D_evo1_plus[-1]+shift))   # trial and error to find a good position
# pl.annotate(r'$c^-$', (len(v)*3/4, -D_evo1_minus[-1]-2*shift))   # trial and error to find a good position
pl.ylim(-.02, .045)
pl.tight_layout()
pl.show()


