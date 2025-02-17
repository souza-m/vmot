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
import uniform_marginals_T2 as exT2, uniform_marginals_T3 as exT3
from matplotlib.patches import Rectangle


# ---------- choose ----------

# first example
d, T = 2, 2
ref_value = 5.1124   # analytical true value Ex. 4.2
title='Numerical value (T = 2)'

# ---------- choose ----------

# second example
d, T = 2, 3
# upper_bound = ref_value
# lower_bound = 2 * (np.e - np.sqrt(np.e)) * (2/3) * (np.e ** 1.5 - 1)   # true value for the independent coupling
title='Numerical value (T = 3)'

# ---------- ------ ----------


# plot and print mean and std
model1, D1_series, H1_series = vmot.load_results(f'uniform_d{d}_T{T}_110_mono_cpu')
model2, D2_series, H2_series = vmot.load_results(f'uniform_d{d}_T{T}_110_full_cpu')
evo1 = np.array(D1_series[100:]) + np.array(H1_series) # random, independent
evo2 = np.array(D2_series[100:]) + np.array(H2_series) # random, monotone
vmot.convergence_plot([evo2, evo1], ['reduced', 'full'], ref_value=ref_value, title='Numerical value (T = 2)')
print(f'mean    mono = {evo1.mean():8.5f}, full = {evo2.mean():8.5f}')
print(f'std     mono = {evo1.std():8.5f}, full = {evo2.std():8.5f}')




# heatmap
def plot_heatmap_T2(heat, x, y):
    s = 4
    c = 'black'
    a = .1
    fig, ax = pl.subplots(1, 2, sharex=True, sharey=True, figsize=[9,4.1])
    ax[0].set_xlim([-.1, 1.6])
    ax[0].add_patch(Rectangle((0.5, 0.5), 0.5, 0.5, ls=':', facecolor="none", ec='k'))
    ax[0].set_aspect('equal', 'box')
    ax[0].set_title('X1 x X2')
    ax[0].scatter(heat['x1'], heat['x2'], s=s, color=c, alpha=a)
    ax[1].set_xlim([-.1, 1.6])
    ax[1].add_patch(Rectangle((0.5, 0.0), 0.5, 1.5, ls=':', facecolor="none", ec='k'))
    ax[1].set_aspect('equal', 'box')
    ax[1].set_title('Y1 x Y2')
    ax[1].scatter(heat['y1'], heat['y2'], s=s, color=c, alpha=a)
    fig.show()

# def plot_heatmap_T2_all(heat, x, y):
#     s = 8
#     c = 'black'
#     a = .1
#     fig, ax = pl.subplots(2, 2, sharex=True, sharey=True, )
#     ax[0,0].set_xlim([0, 1.5])
#     ax[0,0].set_ylim([0, 1.5])
#     ax[0,0].set_title('X1 x X2')
#     ax[0,0].scatter(heat['x1'], heat['x2'], s=s, color=c, alpha=a)
#     ax[0,1].set_xlim([0, 1.5])
#     ax[0,1].set_ylim([0, 1.5])
#     ax[0,1].set_title('X1 x Y1')
#     ax[0,1].scatter(heat['x1'], heat['y1'], s=s, color=c, alpha=a)
#     ax[1,0].set_xlim([0, 1.5])
#     ax[1,0].set_ylim([0, 1.5])
#     ax[1,0].set_title('X2 x Y2')
#     ax[1,0].scatter(heat['x2'], heat['y2'], s=s, color=c, alpha=a)
#     ax[1,1].set_xlim([0, 1.5])
#     ax[1,1].set_ylim([0, 1.5])
#     ax[1,1].set_title('Y1 x Y2')
#     ax[1,1].scatter(heat['y1'], heat['y2'], s=s, color=c, alpha=a)
#     fig.show()

def plot_heatmap_T3(heat, x, y, z):    s = 4
    c = 'black'
    a = .1
    fig, ax = pl.subplots(1, 3, sharex=True, sharey=True, figsize=[9,4.1])
    ax[0].set_xlim([-.1, 1.6])
    # ax[0].set_ylim([0, 1.5])
    ax[0].add_patch(Rectangle((0.5, 0.5), 0.5, 0.5, ls=':', facecolor="none", ec='k'))
    ax[0].set_aspect('equal', 'box')
    ax[0].set_title('X1 x X2')
    ax[0].scatter(heat['x1'], heat['x2'], s=s, color=c, alpha=a)
    # ax[1].set_xlim([0, 1.5])
    ax[1].set_xlim([-.1, 1.6])
    ax[1].add_patch(Rectangle((0.5, 0.0), 0.5, 1.5, ls=':', facecolor="none", ec='k'))
    ax[1].set_aspect('equal', 'box')
    ax[1].set_title('Y1 x Y2')
    ax[1].scatter(heat['y1'], heat['y2'], s=s, color=c, alpha=a)
    fig.show()


# margins based on beta_conjugate
n_points = 1000000

# T = 2
monotone = True
model = model1 if monotone else model2
u, v, x, y = exT2.random_sample(n_points, monotone = monotone)
c = exT2.cost_f(x, y)
ws = vmot.generate_working_sample_T2(u, v, x, y, c)
D, H, c = vmot.mtg_dual_value(ws, model, d, T, monotone)
pi_hat, lbound, ubound = vmot.mtg_numeric_pi_hat(ws, model, d, T, monotone = monotone, opt_parameters = exT2.opt_parameters)
pi_hat.sum()

normalize = True
if normalize:
    pi_hat /= pi_hat.sum()
    
    
    
heatmap = pd.DataFrame({ 'x1'  : x[:,0],
                         'x2'  : x[:,1],
                         'y1'  : y[:,0],
                         'y2'  : y[:,1],
                         'p'   : pi_hat  })

scale = 20. * max(pi_hat)
heat = heatmap.iloc[pi_hat > scale * np.random.random(len(pi_hat))]
print(len(heat))
plot_heatmap_T2(heat, x, y)

heat = heatmap.iloc[:len(heat)]
plot_heatmap_T2(heat, x, y)









# individual plot
_, __, D1_series = vmot.load_results(f'uniform_d{d}_T{T}_{existing_i}_full')
_, __, D2_series = vmot.load_results(f'uniform_d{d}_T{T}_{existing_i}_mono')
evo1 = np.array(D1_series) # random, independent
evo2 = np.array(D2_series) # random, monotone
vmot.convergence_plot([evo2, evo1], ['reduced', 'full'], ref_value=ref_value)


# reference value
ref_value = 5.1124   # true VMOT solution, see Example 4.2 of reference paper

# u, v, x, y = random_sample(n_points=10000000, monotone = True)
# c = cost_f(x, y)
# sample_mean = c.mean()

# heatmap

# vmot.dump_results([model1, D1_series], f'uniform_full_d{d}_T{T}_{existing_i}')
# vmot.dump_results([model2, D2_series], f'uniform_mono_d{d}_T{T}_{existing_i}')
model1, D1_series = vmot.load_results(f'uniform_d{d}_T{T}_{existing_i}_mono')
model2, D2_series = vmot.load_results(f'uniform_d{d}_T{T}_{existing_i}_full')


# D1, H1, pi_star1 = vmot.mtg_dual_value(ws1, model1, d, T, monotone = False, opt_parameters = opt_parameters, normalize_pi = False)
# D2, H2, pi_star2 = vmot.mtg_dual_value(ws2, model2, d, T, monotone = True,  opt_parameters = opt_parameters, normalize_pi = False)
# pi_star1.shape
# pi_star1.sum(axis=0)

# pl.figure()
# pl.plot(pi_star1.sum(axis=0))
# pl.plot(pi_star1.sum(axis=1))
# pi_star1[:,6050:6055]


def plot_heatmap(heat, x, y):
    s = 8
    c = 'black'
    a = .1
    fig, ax = pl.subplots(2, 2, sharex=True, sharey=True, )
    ax[0,0].set_xlim([0, 3.])
    ax[0,0].set_ylim([0, 3.])
    ax[0,0].set_title('X1 x X2')
    ax[0,0].scatter(heat['x1'], heat['x2'], s=s, color=c, alpha=a)
    ax[0,1].set_xlim([0, 3.])
    ax[0,1].set_ylim([0, 3.])
    ax[0,1].set_title('X1 x Y1')
    ax[0,1].scatter(heat['x1'], heat['y1'], s=s, color=c, alpha=a)
    ax[1,0].set_xlim([0, 3.])
    ax[1,0].set_ylim([0, 3.])
    ax[1,0].set_title('X2 x Y2')
    ax[1,0].scatter(heat['x2'], heat['y2'], s=s, color=c, alpha=a)
    ax[1,1].set_xlim([0, 3.])
    ax[1,1].set_ylim([0, 3.])
    ax[1,1].set_title('Y1 x Y2')
    ax[1,1].scatter(heat['y1'], heat['y2'], s=s, color=c, alpha=a)
    fig.show()

# margins based on beta_conjugate
monotone = False
existing_i = 100
mono_label = 'mono' if monotone else 'full'
path = f'uniform_d{d}_T{T}_{existing_i}_' + mono_label
model, opt, D_series = vmot.load_results(path)

u, v, x, y = random_sample(n_points, monotone = monotone)
c = cost_f(x, y)
print(c.mean())
ws = vmot.generate_working_sample_T2(u, v, x, y, c)
D, H, c = vmot.mtg_dual_value(ws, model, d, T, monotone)
pi_hat, lbound, ubound = vmot.mtg_numeric_pi_hat(ws, model, d, T, monotone = False, opt_parameters = opt_parameters)
pi_hat.sum()
heatmap = pd.DataFrame({ 'x1'  : x[:,0],
                          'x2'  : x[:,1],
                          'y1'  : y[:,0],
                          'y2'  : y[:,1],
                          'p'   : pi_hat  })

scale = 20. * max(pi_hat)
heat = heatmap.iloc[pi_hat > scale * np.random.random(len(pi_hat))]
print(len(heat))
plot_heatmap(heat, x, y)

heat = heatmap.iloc[:len(heat)]
plot_heatmap(heat, x, y)

