# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:35:25 2021
@author: souzam
PyTorch implementation of Eckstein and Kupper 2021 - Computation of Optimal Transport...
"""

# Example 1.1: solve cross-product cost with normal marginals, d = 2
# Example 2:   solve cross-product cost with empirical marginals, d = 2
#
#   Cost function (to be maximized):   cost_f(x, y) = y1 * y2
#
# Coupling structure:
#   (1)   independent (4 dimensions))
#   (2)   montone, dimension reduction on x (3 dimensions)
#
# Method: use vmot_core to generate "working sample" objects to be used in the optimization loop.
# The optimization loop performs the dual approximation based on EK21.
# We experiment with dimensionlality reduction and random vs grid sampling methods.


import numpy as np
# import pandas as pd
import matplotlib.pyplot as pl
# from scipy.stats import norm

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
def minus_cost_f(x, y):
    return -cost_f(x, y)


# # example 1 - normal marginals

# # choose scales
# sig1 = 1.0
# sig2 = 1.0
# rho1 = np.sqrt(2.0)
# rho2 = np.sqrt(3.0)
# x_normal_scale = [sig1, sig2]
# y_normal_scale = [rho1, rho2]

# # reference value (see proposition)
# lam1 = np.sqrt(rho1 ** 2 - sig1 ** 2)
# lam2 = np.sqrt(rho2 ** 2 - sig2 ** 2)
# ref_value = (A + B) * sig1 * sig2 + B * lam1 * lam2
# print(f'normal marginals exact solution: {ref_value:8.4f}')

# # inverse cumulatives
# def normal_inv_cum_xi(q, i):
#     return norm.ppf(q) * x_normal_scale[i]

# def normal_inv_cum_yi(q, i):
#     return norm.ppf(q) * y_normal_scale[i]

# def normal_inv_cum_x(q):
#     z = norm.ppf(q)
#     return np.array([z * x_normal_scale[i] for i in range(d)]).T

# # working samples
# ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, normal_inv_cum_xi, normal_inv_cum_yi, cost_f)
# ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, normal_inv_cum_x, normal_inv_cum_yi, cost_f)

# # train/store/load
# # model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.mtg_train(ws1, opt_parameters, monotone = False, verbose = 10)
# # model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.mtg_train(ws2, opt_parameters, monotone = True, verbose = 10)
# # vmot.dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1], 'normal')
# # vmot.dump_results([model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2], 'normal_mono')
# # model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.load_results('normal')
# # model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.load_results('normal_mono')

# # load from iterative high d
# existing_i = 10
# model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.load_results(f'normal_d2_{existing_i}')
# model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.load_results(f'normal_mono_d2_{existing_i}')

# # plot
# evo1 = np.array(D_evo1) # random, independent
# evo2 = np.array(D_evo2) # random, monotone
# vmot.convergence_plot([evo2, evo1], ['monotone', 'independent'], ref_value=ref_value, title='Numerical value convergence - Normal marginals (d=2)')


# example - empirical

I = 20   # iterations
existing_i = 0
np.random.seed(1)

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
uvset1  = vmot.random_uvset(n_points, d)
uvset2  = vmot.random_uvset_mono(n_points, d)
# ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, empirical_inv_cum_xi, empirical_inv_cum_yi, cost_f)
# ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, empirical_inv_cum_x, empirical_inv_cum_yi, cost_f)
ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, empirical_inv_cum_xi, empirical_inv_cum_yi, minus_cost_f)
ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, empirical_inv_cum_x, empirical_inv_cum_yi, minus_cost_f)
cost_series = np.hstack([ws1[:,-2], ws2[:,-2]])
sample_mean_cost = cost_series.mean()   # lower reference for the optimal cost

if existing_i == 0:
    # train/store/load
    print('\niteration 1 (new model)\n')
    model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.mtg_train(ws1, opt_parameters, monotone = False, verbose = 10)
    model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.mtg_train(ws2, opt_parameters, monotone = True, verbose = 10)
    existing_i = 1
    print('models generated')
    vmot.dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1], 'minus_empirical_1')
    vmot.dump_results([model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2], 'minus_empirical_mono_1')
else:
    # load
    print(f'\nloading model {existing_i}')
    model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.load_results(f'minus_empirical_{existing_i}')
    model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.load_results(f'minus_empirical_mono_{existing_i}')


# iterate optimization
while existing_i < I:
    
    # new random sample
    print(f'\niteration {existing_i+1}\n')
    uvset1 = vmot.random_uvset(n_points, d)
    uvset2 = vmot.random_uvset_mono(n_points, d)
    # ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, empirical_inv_cum_xi, empirical_inv_cum_yi, cost_f)
    # ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, empirical_inv_cum_x, empirical_inv_cum_yi, cost_f)
    ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, empirical_inv_cum_xi, empirical_inv_cum_yi, minus_cost_f)
    ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, empirical_inv_cum_x, empirical_inv_cum_yi, minus_cost_f)
    cost_series = np.hstack([cost_series, ws1[:,-2], ws2[:,-2]])
    sample_mean_cost = cost_series.mean()   # lower reference for the optimal cost
    print(f'sample mean cost = {sample_mean_cost:7.4f}')

    _model1, _D_evo1, _H_evo1, _P_evo1, _ds_evo1, _hs_evo1 = vmot.mtg_train(ws1, opt_parameters, model=model1, monotone = False, verbose = 10)
    _model2, _D_evo2, _H_evo2, _P_evo2, _ds_evo2, _hs_evo2 = vmot.mtg_train(ws2, opt_parameters, model=model2, monotone = True, verbose = 10)
    
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
    vmot.dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1], f'minus_empirical_{existing_i}')
    vmot.dump_results([model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2], f'minus_empirical_mono_{existing_i}')
    
    # plot
    evo1 = np.array(D_evo1) # random, independent
    evo2 = np.array(D_evo2) # random, monotone
    vmot.convergence_plot([-evo2, -evo1], ['reduced', 'full'], ref_value=-sample_mean_cost)
    vmot.convergence_plot([-evo2, -evo1], ['reduced', 'full'], ref_value=-sample_mean_cost)
    
    
# report mean and std over a collection of samples
report = False
if report:
    collection_size = 10
    collection_cost_series = None
    D1_series = []
    D2_series = []
    P1_series = []
    P2_series = []
    for i in range(collection_size):
        uvset1 = vmot.random_uvset(n_points, d)
        uvset2 = vmot.random_uvset_mono(n_points, d)
        ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, empirical_inv_cum_xi, empirical_inv_cum_yi, cost_f)
        ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, empirical_inv_cum_x, empirical_inv_cum_yi, cost_f)
        if collection_cost_series is None: 
            collection_cost_series = np.hstack([ws1[:,-2], ws2[:,-2]])
        else:
            collection_cost_series = np.hstack([cost_series, ws1[:,-2], ws2[:,-2]])
        D1, P1, __ = vmot.mtg_dual_value(model1, ws1, opt_parameters, normalize_pi = False)
        D2, P2, __ = vmot.mtg_dual_value(model2, ws2, opt_parameters, normalize_pi = False)
        D1_series.append(D1)
        D2_series.append(D2)
        P1_series.append(P1)
        P2_series.append(P2)
    sample_mean_cost = collection_cost_series.mean()   # lower reference for the optimal cost
    print(f'sample mean cost = {sample_mean_cost:7.4f}')
    print('dual value')
    print(f'full:     mean = {np.mean(D1_series):8.4f};   std = {np.std(D1_series):8.4f}')
    print(f'reduced:  mean = {np.mean(D2_series):8.4f};   std = {np.std(D2_series):8.4f}')
    print('penalty')
    print(f'full:     mean = {np.mean(P1_series):8.4f};   std = {np.std(P1_series):8.4f}')
    print(f'reduced:  mean = {np.mean(P2_series):8.4f};   std = {np.std(P2_series):8.4f}')




# heat empirical
# sets of (u,v) points
grid_n  = 40
uvset1g = vmot.grid_uvset(grid_n, d)
uvset2g = vmot.grid_uvset_mono(grid_n, d)
print('sample shapes')
print('independent grid    ', uvset1g.shape)
print('monotone grid       ', uvset2g.shape)

ws1g, grid1 = vmot.generate_working_sample_uv(uvset1g, empirical_inv_cum_xi, empirical_inv_cum_yi, cost_f)
ws2g, grid2 = vmot.generate_working_sample_uv_mono(uvset2g, empirical_inv_cum_x, empirical_inv_cum_yi, cost_f)
D1, H1, pi_star1 = vmot.mtg_dual_value(model1, ws1g, opt_parameters, normalize_pi = False)
D2, H2, pi_star2 = vmot.mtg_dual_value(model2, ws2g, opt_parameters, normalize_pi = False)






# plot
ref_value = sample_mean_cost
ref_color = 'black'
ref_label = 'lower bound'
labels = ['reduced', 'full']
# title='Convergence - empirical marginals (d = 2)'
evo1 = np.array(D_evo1)[:100]
evo2 = np.array(D_evo2)[:100]

pl.figure(figsize = [5,5])
for v in [evo2, evo1]:
    pl.plot(range(1, len(v)+1), v)
if not ref_value is None:
    pl.axhline(ref_value, linestyle=':', color=ref_color)
pl.legend(labels)
# pl.title(title)
shift = .035 * (pl.gca().get_ylim()[1] - pl.gca().get_ylim()[0])
pl.annotate('(lower bound)', (len(v)*1/5, ref_value-shift), color=ref_color)   # trial and error to find a good position
pl.show()





vmot.convergence_plot([evo1, evo2], ['full', 'reduced'], ref_value=sample_mean_cost, title='Convergence - empirical marginals (d = 2)')

















# pi_star
ws1g, grid1 = vmot.generate_working_sample_uv(uvset1g, empirical_inv_cum_xi, empirical_inv_cum_yi, cost_f)
ws2g, grid2 = vmot.generate_working_sample_uv_mono(uvset2g, empirical_inv_cum_x, empirical_inv_cum_yi, cost_f)
D1, H1, pi_star1 = vmot.mtg_dual_value(model1, ws1g, opt_parameters, normalize_pi = False)
D2, H2, pi_star2 = vmot.mtg_dual_value(model2, ws2g, opt_parameters, normalize_pi = False)

pi_star1.sum()
pi_star2.sum()

heatmap(ws1g, pi_star1)
heatmap(ws2g, pi_star2)


# test mode - reiterate train recycling the model
# note: load correct working samples above before running

# _opt_parameters = { 'penalization'    : 'L2',
#                     'beta_multiplier' : 1,
#                     'gamma'           : 100,
#                     'batch_size'      : 2000,   # no special formula for this
#                     'macro_epochs'    : 1,
#                     'micro_epochs'    : 1      }
_opt_parameters = opt_parameters.copy()

_model1, _D_evo1, _H_evo1, _P_evo1, _ds_evo1, _hs_evo1 = vmot.mtg_train(ws1, _opt_parameters, model = model1, monotone = False, verbose = 100)
_model2, _D_evo2, _H_evo2, _P_evo2, _ds_evo2, _hs_evo2 = vmot.mtg_train(ws2, _opt_parameters, model = model2, monotone = True, verbose = 100)

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

vmot.dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1, model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2], 'test')
# vmot.dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1, model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2], 'normal')
# vmot.dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1, model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2], 'empirical')
# model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1, model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.load_results('test')



# def heatmaps(grid, pi):
#     # generate heatmap matrix
#     X = pd.DataFrame(grid)[[0,1]]
#     X.columns = ['X1', 'X2']
#     X['pi'] = pi
#     X = X.groupby(['X1', 'X2']).sum()
#     heat_x = X.pivot_table(values='pi', index='X1', columns='X2', aggfunc='sum').values
#     heat_x[heat_x==0] = np.nan
    
#     Y= pd.DataFrame(grid)[[2,3]]
#     Y.columns = ['Y1', 'Y2']
#     Y['pi'] = pi
#     Y = Y.groupby(['Y1', 'Y2']).sum()
#     heat_y = Y.pivot_table(values='pi', index='Y1', columns='Y2', aggfunc='sum').values
#     heat_y[heat_y==0] = np.nan
    
#     # plot
#     figsize = [18,8]
#     fig, ax = pl.subplots(1, 3, figsize=figsize)
#     ax[0].imshow(heat_x, cmap = "Reds")
#     im = ax[1].imshow(heat_y, cmap = "Reds")
#     ax[0].invert_yaxis()
#     ax[1].invert_yaxis()
#     fig.colorbar(im)
    
#     return heat_x, heat_y
