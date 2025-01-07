# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:35:25 2021
@author: souzam
PyTorch implementation of Eckstein and Kupper 2019 - Computation of Optimal Transport...
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import vmot_dual as vmot
import matplotlib.pyplot as pl
import matplotlib.colors as mcolors
# from cycler import cycler


# example with cross-product cost and normal marginals, d = 2, T = 3

# consider prices of two assets X, Y at future times 1, 2, 3
# prices are random variables X1, X2, X3, Y1, Y2, Y3
# the distributions of each r.v. is known
# Xi ~ mu_i; Yi ~ nu_i, centered and in convex order as
# X1 <= X2 <= X3 and Y1 <= Y2 <= Y3
# in this example, we consider normal distributions
# our goal is to calculate the upper and lower bounds of the price of a contract
# that pays c(X,Y) = X3 * Y3
#
# (I)  max E[ c(X,Y) ]
# (II) min E[ c(X,Y) ]
#
# over all possible joint distributions of (X,Y)


# two marginals, three periods
# full problem has 6 dimensions
# reduced problem has 5 dimensions
d = 2
n_periods = 3

# fixed variance parameters
# each sequence of variances has to be increasing to guarantee convex order

np.random.seed(1)
mu = [np.trunc(10 * np.random.random()) / 10 + i + 1 for i in range(3)]
nu = [np.trunc(10 * np.random.random()) / 10 + i + 1 for i in range(3)]
print(f'X1 ~ N(0, {mu[0]}),  X2 ~ N(0, {mu[1]}),  X3 ~ N(0, {mu[2]})')
print(f'Y1 ~ N(0, {nu[0]}),  Y2 ~ N(0, {nu[1]}),  Y3 ~ N(0, {nu[2]})')


# inverse cumulatives
def normal_inv_cum_xi(q, i):
    return norm.ppf(q) * mu[i]

def normal_inv_cum_yi(q, i):
    return norm.ppf(q) * nu[i]

def normal_inv_cum_x(q):
    return norm.ppf(q) * np.array([mu[0], nu[0]])


# generate synthetic sample
# sample includes quantile points (u,v) and corresponding (x,y) points
# if mono_coupled, u = (u1, u2, u3) and v = (v2, v3), since v1 = u1, therefore (u,v) is in [0,1]^5
# otherwise, u = (u1, u2, u3) and v = (v1, v2, v3), and (u,v)  is in [0,1]^6
# (x,y) = (x1, x2, x3, y1, y2, y3) is in R^6
def random_sample(n_points, monotone):
    u = np.random.random((n_points, 3))
    x = np.array([normal_inv_cum_xi(u[:,i], i) for i in range(3)]).T
    if monotone:
        v = np.random.random((n_points, 2))
        y = np.array([normal_inv_cum_yi(u[:,0], 0), normal_inv_cum_yi(v[:,0], 1), normal_inv_cum_yi(v[:,1], 2)]).T
    else:
        v = np.random.random((n_points, 3))
        y = np.array([normal_inv_cum_xi(v[:,i], i) for i in range(3)]).T
    return u, v, x, y
    


# optimization parameters
opt_parameters = { 'penalization'    : 'L2',    # penalization shape
                   'beta_multiplier' : 1,       # penalization multiplier
                   'gamma'           : 1000,    # penalization parameter
                   'epochs'          : 10,      # iteration parameter
                   'batch_size'      : 2000  }  # iteration parameter  


# cost function
def cost_f(x, y):
    return x[:,2] * y[:,2]


# reference value (see formula in proposition (?))
# TO BE COMPLETED
# lam = np.sqrt(rho ** 2 - sig ** 2)
ref_value = 0.0
# for i in range(0, d):
#     for j in range(i+1, d):
#         ref_value = ref_value + (A[i,j] + B[i,j]) * sig[i] * sig[j] + B[i,j] * lam[i] * lam[j]
# print(f'exact solution: {ref_value:8.4f}')



# --- process batches and save models (takes long time) ---

# batch control
I = 30            # total desired iterations
existing_i = 0    # last iteration saved
n_points = 100000 # sample points at each iteration

if existing_i == 0:
    # new random sample
    print('\niteration 1 (new model)\n')
    
    # regular coupling
    u, v, x, y = random_sample(n_points, monotone = False)
    c = cost_f(x, y)
    ws1 = vmot.generate_working_sample(u, v, x, y, c)
    
    # monotone coupling
    u, v, x, y = random_sample(n_points, monotone = True)
    c = cost_f(x, y)
    ws2 = vmot.generate_working_sample(u, v, x, y, c)
    print('samples generated, shapes ', ws1.shape, 'and', ws2.shape)
    
    # models
    model1 = vmot.generate_model(d, n_periods, monotone = False) 
    model2 = vmot.generate_model(d, n_periods, monotone = True) 
    print('models generated')

    # train
    D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.mtg_train(ws1, model1, d, n_periods, monotone = False, opt_parameters=opt_parameters, verbose = 10)
    D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.mtg_train(ws2, model2, d, n_periods, monotone = True,  opt_parameters=opt_parameters, verbose = 10)

    # store models and evolution
    existing_i = 1
    vmot.dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1], f'portfolio_normal_full_d{d}_{existing_i}')
    vmot.dump_results([model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2], f'portfolio_normal_mono_d{d}_{existing_i}')
    
else:
    
    # load existing model
    print(f'\nloading model {existing_i}')
    model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.load_results(f'portfolio_normal_full_d{d}_{existing_i}')
    model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.load_results(f'portfolio_normal_mono_d{d}_{existing_i}')
    
# iterative parsing
while existing_i < I:
    
    # regular coupling
    u, v, x, y = random_sample(n_points, monotone = False)
    c = cost_f(x, y)
    ws1 = vmot.generate_working_sample_uv(u, v, x, y, c)
    
    # monotone coupling
    u, v, x, y = random_sample(n_points, monotone = True)
    c = cost_f(x, y)
    ws2 = vmot.generate_working_sample_uv(u, v, x, y, c)
    print('samples generated')
    
    _D_evo1, _H_evo1, _P_evo1, _ds_evo1, _hs_evo1 = vmot.mtg_train(ws1, model1, d, n_periods, monotone = False, opt_parameters=opt_parameters, verbose = 10)
    _D_evo2, _H_evo2, _P_evo2, _ds_evo2, _hs_evo2 = vmot.mtg_train(ws2, model2, d, n_periods, monotone = True,  opt_parameters=opt_parameters, verbose = 10)
    print('models updated')
    # CHECK if models are being updated 
    
    # sequential storage of the variables' evolution
    D_evo1  = D_evo1  + _D_evo1
    H_evo1  = H_evo1  + _H_evo1
    P_evo1  = P_evo1  + _P_evo1
    ds_evo1 = ds_evo1 + _ds_evo1
    hs_evo1 = hs_evo1 + _hs_evo1
    
    D_evo2  = D_evo2  + _D_evo2
    H_evo2  = H_evo2  + _H_evo2
    P_evo2  = P_evo2  + _P_evo2
    ds_evo2 = ds_evo2 + _ds_evo2
    hs_evo2 = hs_evo2 + _hs_evo2
    
    existing_i += 1
    vmot.dump_results([model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1], f'portfolio_normal_full_d{d}_{existing_i}')
    vmot.dump_results([model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2], f'portfolio_normal_mono_d{d}_{existing_i}')

        
# individual plot
evo1 = np.array(D_evo1) # random, independent
evo2 = np.array(D_evo2) # random, monotone
vmot.convergence_plot([evo2, evo1], ['reduced', 'full'], ref_value=ref_value)


# report mean and std over a collection of samples created during training
for o, dual_series in enumerate([D_evo1, D_evo2]):
    print()
    print('full' if o == 0 else 'mono')
    print(f'tail mean {np.mean(dual_series[200:]):8.4f}')
    print(f'tail std  {np.std(dual_series[200:]):8.4f}')


# heat map
# step 1: load cost function and inverse cumulative functions (run the main "process" block above with d=2)
# step 2:
grid_n  = 32
uvset1g = vmot.grid_uvset(grid_n, d)
uvset2g = vmot.grid_uvset_mono(grid_n, d)

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

# import matplotlib.cm as cm
# utils - heat map
# cmap in collor pattern of the first cathegorical color '#348ABD' or #A60628
colors = ['white', '#A60628']
# colors = ['white', 'red']
# colors = ['white', '#348ABD']
positions = [0, 1]
cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)))
def heatmap(grid, pi, uplim=0):
    # generate heatmap matrix
    X = pd.DataFrame(grid)[[0,1]]
    X.columns = ['X1', 'X2']
    X['pi'] = pi
    X = X.groupby(['X1', 'X2']).sum()
    heat = X.pivot_table(values='pi', index='X1', columns='X2', aggfunc='sum').values
    heat[heat==0] = np.nan

    # plot
    # figsize = [8,5]
    figsize = [5,3]
    fig, ax = pl.subplots(figsize=figsize)
    # im = ax.imshow(heat, cmap='Reds', extent=[0,1,1,0])
    im = ax.imshow(heat, cmap=cmap, extent=[0,1,1,0])
    
    # keep consistency between x and y scales
    if uplim == 0:
        uplim = np.nanmax(heat)
    im.set_clim(0, uplim)
    
    ax.set_xlabel('U1')
    ax.set_ylabel('U2')
    ax.invert_yaxis()
    ax.figure.colorbar(im)
    fig.tight_layout()
    
    return heat    
# heat = vmot.heatmap(grid1[:,:2], pi_star1)   # X, independent
heat = heatmap(grid2[:,:2], pi_star2)   # X, monotone
heat = heatmap(grid1[:,:2], pi_star1, uplim=np.nanmax(heat))   # X, independent, sharing color scale with monotone



# check diagonal (test mode)
heat_marginal = np.nansum(heat, axis=0)
fig, ax = pl.subplots()
ax.set_ylim(0, max(heat_marginal))
ax.plot(heat_marginal)

pl.figure()
pl.plot(pi_star2)


# tentative to avoid editting
# generate heatmap matrix
X = pd.DataFrame(grid1)[[0,1]]
X.columns = ['X1', 'X2']
X['pi'] = pi_star1
X = X.groupby(['X1', 'X2']).sum()
heat1 = X.pivot_table(values='pi', index='X1', columns='X2', aggfunc='sum').values
heat1[heat1==0] = np.nan

X = pd.DataFrame(grid2)[[0,1]]
X.columns = ['X1', 'X2']
X['pi'] = pi_star2
X = X.groupby(['X1', 'X2']).sum()
heat2 = X.pivot_table(values='pi', index='X1', columns='X2', aggfunc='sum').values
heat2[heat2==0] = np.nan

# plot
# figsize = [8,5]
figsize = [6.85,3]
fig, (ax1, ax2) = pl.subplots(1, 2, figsize=figsize, sharey=True)
# im = ax.imshow(heat, cmap='Reds', extent=[0,1,1,0])
im1 = ax1.imshow(heat1, cmap=cmap, extent=[0,1,1,0])
im2 = ax2.imshow(heat2, cmap=cmap, extent=[0,1,1,0])

# keep consistency between x and y scales
uplim = np.nanmax(heat2)
im1.set_clim(0, uplim)

ax1.set_xlabel('U1')
ax1.set_ylabel('U2')
ax1.invert_yaxis()
ax2.set_xlabel('U1')
# ax2.invert_yaxis()
ax2.figure.colorbar(im2)

fig.tight_layout()
fig.show()