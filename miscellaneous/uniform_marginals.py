# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:35:25 2021
@author: souzam
PyTorch implementation of Eckstein and Kupper 2019 - Computation of Optimal Transport...
"""

import numpy as np
import pandas as pd
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

# --- uniform and convoluted marginals ---
#
# # cumulative of U[-1, 1]
# def F_inv0(x):
#     return 2 * x - 1.

# # cumulative of U[-1, 1] # U[-1, 1]
# def F_inv_a(x):
#     return 4 * np.sqrt(x)
    
# def F_inv_b(x):
#     return 4 * x
    
# def F_inv_c(x):
#     return -4 * np.sqrt(1. - x)
    
# def F_inv1(x):
#     return F_inv_a(np.minimum(x, 1/4)) +\
#            F_inv_b(np.maximum(np.minimum(x, 3/4), 1/4)) +\
#            F_inv_c(np.maximum(x,  3/4)) - 2.


# --- uniform marginals ---

# inverse cumulative of U[0.5 , 1.0]
def F_inv0(u):
    return 0.5 * (u + 1.) 

# inverse cumulative of U[0.0 , 1.5]
def F_inv1(u):
    return 1.5 * u

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
    return np.exp(x[:,0] * y[:,0]) + np.exp(x[:,1] * y[:,1]) + np.exp(x[:,2] * y[:,2])




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