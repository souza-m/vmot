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


# example with cross-product cost and uniform marginals, d = 2, T = 3

# over all possible joint distributions of (X,Y)


# two marginals, three periods
# full problem has 6 dimensions
# reduced problem has 5 dimensions
d = 2
nperiods = 3

# fixed variance parameters
# each sequence of variances has to be increasing to guarantee convex order

# cumulative of U[-1, 1]
def F_inv0(x):
    return 2 * x - 1.

# cumulative of U[-1, 1] # U[-1, 1]
def F_inv_a(x):
    return 4 * np.sqrt(x)
    
def F_inv_b(x):
    return 4 * x
    
def F_inv_c(x):
    return -4 * np.sqrt(1. - x)
    
def F_inv1(x):
    return F_inv_a(np.minimum(x, 1/4)) +\
           F_inv_b(np.maximum(np.minimum(x, 3/4), 1/4)) +\
           F_inv_c(np.maximum(x,  3/4)) - 2.

# x = np.linspace(0., 1., 201)
# pl.figure()
# pl.plot(x, F0(x))
# pl.plot(x, F1(x))
# pl.axhline(0., linestyle=':', color='black')

F_inv_x = [F_inv0, F_inv0, F_inv1]
F_inv_y = [F_inv0, F_inv1, F_inv1]


# generate synthetic sample
# sample includes quantile points (u,v) and corresponding (x,y) points
# if mono_coupled, u = (u1, u2, u3) and v = (v2, v3), since v1 = u1, therefore (u,v) is in [0,1]^5
# otherwise, u = (u1, u2, u3) and v = (v1, v2, v3), and (u,v)  is in [0,1]^6
# (x,y) = (x1, x2, x3, y1, y2, y3) is in R^6
def random_sample(n_points, monotone):
    u = np.random.random((n_points, 3))
    x = np.array([F_inv_x[i](u[:,i]) for i in range(3)]).T
    if monotone:
        v = np.random.random((n_points, 2))   # v0 is replaced by u0
        y = np.array([F_inv_y[0](u[:,0]), F_inv_y[1](v[:,0]), F_inv_y[2](v[:,1])]).T
    else:
        v = np.random.random((n_points, 3))
        y = np.array([F_inv_y[i](v[:,i]) for i in range(3)]).T
    return u, v, x, y
    

# cost function
def cost_f(x, y):
    return x[:,2] * y[:,2]


# reference value (see formula in proposition (?))
# TO BE COMPLETED
# lam = np.sqrt(rho ** 2 - sig ** 2)
ref_value = 3.   # strict upper bound
# for i in range(0, d):
#     for j in range(i+1, d):
#         ref_value = ref_value + (A[i,j] + B[i,j]) * sig[i] * sig[j] + B[i,j] * lam[i] * lam[j]
# print(f'exact solution: {ref_value:8.4f}')





# --- process batches and save models (takes long time) ---

# optimization parameters
opt_parameters = { 'penalization'    : 'L2',    # penalization shape
                   'beta_multiplier' : 1,       # penalization multiplier
                   'gamma'           : 1000,    # penalization parameter
                   'epochs'          : 1,      # iteration parameter
                   'batch_size'      : 2000  }  # iteration parameter  

# batch control
I = 100           # total desired iterations
existing_i = 0    # last iteration saved
n_points = 200000 # sample points at each iteration

if existing_i == 0:
    # new random sample
    print('\niteration 1 (new model)\n')
    
    # regular coupling
    u, v, x, y = random_sample(n_points, monotone = False)
    c = cost_f(x, y)
    ws1 = vmot.generate_working_sample(u, v, x, y, c)                # u1, u2, u3, v1, v2, v3, dif_x12, dif_x23, dif_y12, dif_y23, c, w
    
    # monotone coupling
    u, v, x, y = random_sample(n_points, monotone = True)
    c = cost_f(x, y)
    ws2 = vmot.generate_working_sample(u, v, x, y, c)                # u1, u2, u3, v2, v3, dif_x12, dif_x23, dif_y12, dif_y23, c, w
    print('samples generated, shapes ', ws1.shape, 'and', ws2.shape)
    
    # models
    model1, opt1 = vmot.generate_model(d, nperiods, monotone = False) 
    model2, opt2 = vmot.generate_model(d, nperiods, monotone = True) 
    print('models generated')
    
    
    # train
    D1_series = vmot.mtg_train(ws1, model1, opt1, d, nperiods, monotone = False, opt_parameters=opt_parameters, verbose = 1)
    D2_series = vmot.mtg_train(ws2, model2, opt2, d, nperiods, monotone = True,  opt_parameters=opt_parameters, verbose = 1)

    # store models and evolution
    existing_i = 1
    vmot.dump_results([model1, opt1, D1_series], f'uniform_full_d{d}_T{nperiods}_{existing_i}')
    vmot.dump_results([model2, opt2, D2_series], f'uniform_mono_d{d}_T{nperiods}_{existing_i}')
    
# iterative parsing
while existing_i < I:
    
    # new random sample
    print(f'\niteration {existing_i+1}\n')
    
    # regular coupling
    u, v, x, y = random_sample(n_points, monotone = False)
    c = cost_f(x, y)
    ws1 = vmot.generate_working_sample(u, v, x, y, c)                # u1, u2, u3, v1, v2, v3, dif_x12, dif_x23, dif_y12, dif_y23, c, w
    
    # monotone coupling
    u, v, x, y = random_sample(n_points, monotone = True)
    c = cost_f(x, y)
    ws2 = vmot.generate_working_sample(u, v, x, y, c)                # u1, u2, u3, v2, v3, dif_x12, dif_x23, dif_y12, dif_y23, c, w
    print('samples generated, shapes ', ws1.shape, 'and', ws2.shape)
    
    # load existing model
    print(f'\nloading model {existing_i}')
    model1, opt1, D1_series = vmot.load_results(f'uniform_full_d{d}_T{nperiods}_{existing_i}')
    model2, opt2, D2_series = vmot.load_results(f'uniform_mono_d{d}_T{nperiods}_{existing_i}')
    
    # train
    _D1_series = vmot.mtg_train(ws1, model1, opt1, d, nperiods, monotone = False, opt_parameters=opt_parameters, verbose = 1)
    _D2_series = vmot.mtg_train(ws2, model2, opt2, d, nperiods, monotone = True,  opt_parameters=opt_parameters, verbose = 1)
    existing_i += 1
    print('models updated')
    
    # sequential storage of the variables' evolution
    D1_series = D1_series + _D1_series
    D2_series = D2_series + _D2_series
    
    vmot.dump_results([model1, opt1, D1_series], f'uniform_full_d{d}_T{nperiods}_{existing_i}')
    vmot.dump_results([model2, opt2, D2_series], f'uniform_mono_d{d}_T{nperiods}_{existing_i}')


# individual plot
evo1 = np.array(D1_series) # random, independent
evo2 = np.array(D2_series) # random, monotone
vmot.convergence_plot([evo2, evo1], ['reduced', 'full'], ref_value=ref_value)


# heatmap

D1, H1, pi_star1 = vmot.mtg_dual_value(ws1, model1, d, nperiods, monotone = False, opt_parameters = opt_parameters, normalize_pi = False)
D2, H2, pi_star2 = vmot.mtg_dual_value(ws2, model2, d, nperiods, monotone = True,  opt_parameters = opt_parameters, normalize_pi = False)
pi_star1.shape
pi_star1.sum(axis=0)

pl.figure()
pl.plot(pi_star1.sum(axis=0))
pl.plot(pi_star1.sum(axis=1))
pi_star1[:,6050:6055]








# *** test mode ***
working_sample = ws1[:100,:]
model = model1
d = 2
T = 3
monotone = False
opt_parameters = opt_parameters
verbose = 1
# ***
    
    
    
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