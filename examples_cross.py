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
import pandas as pd
import matplotlib.pyplot as pl
from scipy.stats import norm

import vmot
import option_implied_inverse_cdf as empirical


# processing parameters
d = 2
n_points = 2000000
print(f'd: {d}')
print(f'sample size: {n_points}')
opt_parameters = { 'penalization'    : 'L2',
                   'beta_multiplier' : 1,
                   'gamma'           : 100,
                   'batch_size'      : 2000,   # no special formula for this
                   'epochs'          : 60      }

# cost function to be maximized
A = 0
B = 1
def cost_f(x, y):
    # cost = A.x1.x2 + B.y1.y2
    return A * x[:,0] * x[:,1] + B * y[:,0] * y[:,1]

# sets of (u,v) points
grid_n  = 50
uvset1  = vmot.random_uvset(n_points, d)
uvset2  = vmot.random_uvset_mono(n_points, d)
uvset1g = vmot.grid_uvset(grid_n, d)
# uvset2g = vmot.grid_uvset_mono(int(n**(2*d/(d+1))), d)
uvset2g = vmot.grid_uvset_mono(grid_n, d)
print('sample shapes')
print('independent random  ', uvset1.shape)
print('monotone random     ', uvset2.shape)
print('independent grid    ', uvset1g.shape)
print('monotone grid       ', uvset2g.shape)


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
print(f'normal marginals exact solution: {ref_value:8.4f}')

# inverse cumulatives
def normal_inv_cum_xi(q, i):
    return norm.ppf(q) * x_normal_scale[i]

def normal_inv_cum_yi(q, i):
    return norm.ppf(q) * y_normal_scale[i]

def normal_inv_cum_x(q):
    z = norm.ppf(q)
    return np.array([z * x_normal_scale[i] for i in range(d)]).T

# working samples
ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, normal_inv_cum_xi, normal_inv_cum_yi, cost_f)
ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, normal_inv_cum_x, normal_inv_cum_yi, cost_f)

# train/store/load
# model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.mtg_train(ws1, opt_parameters, monotone = False, verbose = 10)
# model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.mtg_train(ws2, opt_parameters, monotone = True, verbose = 10)
# vmot.dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1], 'normal')
# vmot.dump_results([model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2], 'normal_mono')
model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.load_results('normal')
model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.load_results('normal_mono')

# plot
evo1 = np.array(D_evo1) # random, independent
evo2 = np.array(D_evo2) # random, monotone
vmot.convergence_plot([evo2, evo1], ['monotone', 'independent'], ref_value=ref_value, title='Numerical value convergence - Normal marginals (d=2)')

# pi star, using grid samples
ws1g, grid1 = vmot.generate_working_sample_uv(uvset1g, normal_inv_cum_xi, normal_inv_cum_yi, cost_f)
ws2g, grid2 = vmot.generate_working_sample_uv_mono(uvset2g, normal_inv_cum_x, normal_inv_cum_yi, cost_f)
D1, H1, pi_star1 = vmot.mtg_dual_value(model1, ws1g, opt_parameters, normalize_pi = False)
D2, H2, pi_star2 = vmot.mtg_dual_value(model2, ws2g, opt_parameters, normalize_pi = False)

pi_star1.sum()
pi_star2.sum()
pi_star1 = pi_star1 / pi_star1.sum()
pi_star2 = pi_star2 / pi_star2.sum()

# vmot.plot_sample_2d(ws1g, label='ws1', w=pi_star1, random_sample_size=100000)
# vmot.plot_sample_2d(ws1g, label='ws2', w=pi_star2, random_sample_size=100000)

def heatmap(grid, pi, uplim=0):
    # generate heatmap matrix
    X = pd.DataFrame(grid)[[0,1]]
    X.columns = ['X1', 'X2']
    X['pi'] = pi
    X = X.groupby(['X1', 'X2']).sum()
    heat = X.pivot_table(values='pi', index='X1', columns='X2', aggfunc='sum').values
    heat[heat==0] = np.nan
    
    # plot
    figsize = [10,8]
    fig, ax = pl.subplots(figsize=figsize)
    im = ax.imshow(heat, cmap = "Reds")
    if uplim > 0:
        im.set_clim(0, uplim)
    ax.invert_yaxis()
    ax.figure.colorbar(im)
    
    return heat
    
heat = heatmap(grid1[:,:2], pi_star1)   # X, independent
heat = heatmap(grid1[:,2:], pi_star1, uplim=np.nanmax(heat))   # Y, independent
heat = heatmap(grid2[:,:2], pi_star2)   # X, monotone
heat = heatmap(grid2[:,2:], pi_star2)   # Y, monotone

heat_marginal = np.nansum(heat, axis=0)
fig, ax = pl.subplots()
ax.set_ylim(0, max(heat_marginal))
ax.plot(heat_marginal)


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
ws1, xyset1 = vmot.generate_working_sample_uv(uvset1, empirical_inv_cum_xi, empirical_inv_cum_yi, cost_f)
ws2, xyset2 = vmot.generate_working_sample_uv_mono(uvset2, empirical_inv_cum_x, empirical_inv_cum_yi, cost_f)
sample_mean_cost = 0.5 * (ws1[:,-2].mean() + ws2[:,-2].mean())   # lower reference for the optimal cost

# train/store/load
# model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.mtg_train(ws1, opt_parameters, monotone = False, verbose = 10)
# model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.mtg_train(ws2, opt_parameters, monotone = True, verbose = 10)
# vmot.dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1], 'empirical')
# vmot.dump_results([model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2], 'empirical_mono')
model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.load_results('empirical')
model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.load_results('empirical_mono')

# plot
evo1 = np.array(D_evo1)
evo2 = np.array(D_evo2)
# h1 = np.array(H_evo1)   # random, independent
# h2 = np.array(H_evo2)   # random, monotone
vmot.convergence_plot([evo2, evo1], ['monotone', 'independent'],
                      ref_value=sample_mean_cost, ref_label='lower reference',
                      title='Numerical value convergence - Empirical marginals (d=2)')

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
