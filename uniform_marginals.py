# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:35:25 2021
@author: souzam
PyTorch implementation of Eckstein and Kupper 2019 - Computation of Optimal Transport...
"""

import numpy as np
import pandas as pd
# from scipy.stats import norm
import vmot_dual as vmot
import matplotlib.pyplot as pl
import matplotlib.colors as mcolors
# from cycler import cycler


# example with cross-product cost and uniform marginals, d = 2, T = 3

# over all possible joint distributions of (X,Y)


# two marginals 1, 2
# three periods X, Y, Z
d = 2
T = 3


# inverse cumulative of U[0.5 , 1.0]
def F_inv_a(u):
    return 0.5 * (u + 1.) 

# inverse cumulative of U[0.0 , 1.5]
def F_inv_b(u):
    return 1.5 * u

# inverse cumulative of marginals
F_inv = { 'X1':F_inv_a,
          'X2':F_inv_a,
          'Y1':F_inv_a,
          'Y2':F_inv_b,
          'Z1':F_inv_b,
          'Z2':F_inv_b  }

# generate synthetic sample
# u maps to x, v maps to y, w maps to z through the inverse cumulatives
def random_sample(n_points, monotone):
    u = np.random.random((n_points, 1 if monotone else 2))
    v = np.random.random((n_points, 2))
    w = np.random.random((n_points, 2))
    
    if monotone:
        x = np.array([F_inv['X1'](u[:,0]), F_inv['X2'](u[:,0])]).T
    else:
        x = np.array([F_inv['X1'](u[:,0]), F_inv['X2'](u[:,1])]).T
    y = np.array([F_inv['Y1'](v[:,0]), F_inv['Y2'](v[:,1])]).T
    z = np.array([F_inv['Z1'](w[:,0]), F_inv['Z2'](w[:,1])]).T
                  
    return u, v, w, x, y, z
    

# cost function
def cost_f(x, y, z):
    return np.exp(x[:,0] + x[:,1]) + np.exp(y[:,0] + y[:,1]) + np.exp(z[:,0] + z[:,1])


# reference value
ref_value = 14.43487   # strict upper bound


# --- process batches and save models (takes long time) ---

# optimization parameters
opt_parameters = { 'penalization'    : 'L2',    # penalization shape
                   'beta_multiplier' : 1,       # penalization multiplier
                   'gamma'           : 1000,    # penalization parameter
                   'epochs'          : 1,       # iteration parameter
                   'batch_size'      : 1000  }  # iteration parameter  

# batch control
I = 100           # total desired iterations
existing_i = 0    # last iteration saved
n_points = 1000000 # sample points at each iteration

if existing_i == 0:
    # new random sample
    print('\niteration 1 (new model)\n')
    
    # regular coupling
    u, v, w, x, y, z = random_sample(n_points, monotone = False)
    c = cost_f(x, y, z)
    ws1 = vmot.generate_working_sample_T3(u, v, w, x, y, z, c)
    
    # monotone coupling
    u, v, w, x, y, z = random_sample(n_points, monotone = True)
    c = cost_f(x, y, z)
    ws2 = vmot.generate_working_sample_T3(u, v, w, x, y, z, c)
    print('samples generated, shapes ', ws1.shape, 'and', ws2.shape)
    
    # models
    model1, opt1 = vmot.generate_model(d, T, monotone = False) 
    model2, opt2 = vmot.generate_model(d, T, monotone = True) 
    print('models generated')
    
    # train
    D1_series = vmot.mtg_train(ws1, model1, opt1, d, T, monotone = False, opt_parameters=opt_parameters, verbose = 1)
    D2_series = vmot.mtg_train(ws2, model2, opt2, d, T, monotone = True,  opt_parameters=opt_parameters, verbose = 1)

    # store models and evolution
    existing_i = 1
    vmot.dump_results([model1, opt1, D1_series], f'uniform_full_d{d}_T{T}_{existing_i}')
    vmot.dump_results([model2, opt2, D2_series], f'uniform_mono_d{d}_T{T}_{existing_i}')
    
# iterative parsing
while existing_i < I:
    
    # new random sample
    print(f'\niteration {existing_i+1}\n')
    
    # regular coupling
    u, v, w, x, y, z = random_sample(n_points, monotone = False)
    c = cost_f(x, y, z)
    ws1 = vmot.generate_working_sample_T3(u, v, w, x, y, z, c)
    
    # monotone coupling
    u, v, w, x, y, z = random_sample(n_points, monotone = True)
    c = cost_f(x, y, z)
    ws2 = vmot.generate_working_sample_T3(u, v, w, x, y, z, c)
    print('samples generated, shapes ', ws1.shape, 'and', ws2.shape)
    
    # load existing model
    print(f'\nloading model {existing_i}')
    model1, opt1, D1_series = vmot.load_results(f'uniform_full_d{d}_T{T}_{existing_i}')
    model2, opt2, D2_series = vmot.load_results(f'uniform_mono_d{d}_T{T}_{existing_i}')
    
    # train
    _D1_series = vmot.mtg_train(ws1, model1, opt1, d, T, monotone = False, opt_parameters=opt_parameters, verbose = 1)
    _D2_series = vmot.mtg_train(ws2, model2, opt2, d, T, monotone = True,  opt_parameters=opt_parameters, verbose = 1)
    existing_i += 1
    print('models updated')
    
    # sequential storage of the variables' evolution
    D1_series = D1_series + _D1_series
    D2_series = D2_series + _D2_series
    
    vmot.dump_results([model1, opt1, D1_series], f'uniform_full_d{d}_T{T}_{existing_i}')
    vmot.dump_results([model2, opt2, D2_series], f'uniform_mono_d{d}_T{T}_{existing_i}')


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