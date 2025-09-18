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


folder='gamma_1000/'
existing_i = 120
opt_parameters = { 'gamma'           : 50,      # penalization parameter
                   'epochs'          : 1,       # iteration parameter
                   'batch_size'      : 1000  }  # iteration parameter  

# ---------- choose ----------

# first example
d, T = 2, 2
OT_value = (np.e ** 0.5) * (np.e ** 2 - 1.) / 2   # analytical true value Ex. 4.2
VMOT_value = 5.1124   # analytical true value Ex. 4.2
title_T2 = 'Numerical value (T = 2)'

# ---------- choose ----------

# second example
d, T = 2, 3
# upper_bound = ref_value
# lower_bound = 2 * (np.e - np.sqrt(np.e)) * (2/3) * (np.e ** 1.5 - 1)   # true value for the independent coupling
title_T3 = 'Numerical value (T = 3)'

# ---------- ------ ----------


# historical series
model1, D1_series, H1_series, P1_series = vmot.load_results(folder=folder, label=f'uniform_d{d}_T{T}_{existing_i}_full_cpu')
model2, D2_series, H2_series, P2_series = vmot.load_results(folder=folder, label=f'uniform_d{d}_T{T}_{existing_i}_mono_cpu')
evo1 = np.array(D1_series) + np.array(H1_series) # random, independent
evo2 = np.array(D2_series) + np.array(H2_series) # random, monotone
print(f'mean    full = {evo1.mean():8.5f}, mono = {evo2.mean():8.5f}')
print(f'std     full = {evo1.std():8.5f}, mono = {evo2.std():8.5f}')
# value_series_T2 = [evo1.copy(), evo2.copy()]   # choose
value_series_T3 = [evo1.copy(), evo2.copy()]   # choose


labels = ['Full dimension', 'Reduced dimension', 'OT value', 'VMOT value (true)']
OT_value = (np.e ** 0.5) * (np.e ** 2 - 1.) / 2   # analytical true value Ex. 4.2
VMOT_value = 5.1124   # analytical true value Ex. 4.2
ref_color='black'
ref_label='reference'


fig, ax = pl.subplots(1, 2, sharex=True, sharey=True, figsize=[9,4])

# T = 2
for v in value_series_T2:
    ax[0].plot(range(81, len(v)+81), v[-20:])
ax[0].set_xlim([100, 121])
ax[0].axhline(OT_value, linestyle=':', color='black')
ax[0].axhline(VMOT_value, linestyle=':', color='red')
ax[0].legend(labels, loc='best', bbox_to_anchor=(1., 0.5, 0.0, 0.0))
for v in value_series_T2:
    ax[0].scatter(range(101, len(v)+101), v, s=5)
ax[0].set_xlabel('Iteration')
ax[0].set_title(title_T2)

# T = 3
for v in value_series_T3:
    ax[1].plot(range(81, len(v)+81), v[-20:])
ax[1].axhline(OT_value, linestyle=':', color='black')
ax[1].legend(labels[:-1], loc='best', bbox_to_anchor=(1., 0.5, 0.0, 0.0))
for v in value_series_T3:
    ax[1].scatter(range(81, len(v)+81), v, s=5)
ax[1].set_xlabel('Iteration')
ax[1].set_title(title_T3)

fig.show()






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

def plot_heatmap_T3(heat, x, y, z):
    s = 4
    c = 'black'
    a = .1
    fig, ax = pl.subplots(1, 3, sharex=True, sharey=True, figsize=[13.9,4.1])
    ax[0].set_xlim([-.1, 1.6])
    ax[0].add_patch(Rectangle((0.6, 0.7), 0.3, 0.1, ls=':', facecolor="none", ec='k'))
    ax[0].set_aspect('equal', 'box')
    ax[0].set_title('X1 x X2')
    ax[0].scatter(heat['x1'], heat['x2'], s=s, color=c, alpha=a)
    ax[1].set_xlim([-.1, 1.6])
    ax[1].add_patch(Rectangle((0.5, 0.5), 0.5, 0.5, ls=':', facecolor="none", ec='k'))
    ax[1].set_aspect('equal', 'box')
    ax[1].set_title('Y1 x Y2')
    ax[1].scatter(heat['y1'], heat['y2'], s=s, color=c, alpha=a)
    ax[2].set_xlim([-.1, 1.6])
    ax[2].add_patch(Rectangle((0.5, 0.0), 0.5, 1.5, ls=':', facecolor="none", ec='k'))
    ax[2].set_aspect('equal', 'box')
    ax[2].set_title('Z1 x Z2')
    ax[2].scatter(heat['z1'], heat['z2'], s=s, color=c, alpha=a)
    fig.show()


# margins based on beta_conjugate
n_points = 1000000



# T = 2
monotone = True   # choose
model = model2 if monotone else model1
u, v, x, y = exT2.random_sample(n_points, monotone = monotone)
c = exT2.cost_f(x, y)
ws = vmot.generate_working_sample_T2(u, v, x, y, c)

# T = 3
monotone = False   # choose
model = model2 if monotone else model1
u, v, w, x, y, z = exT3.random_sample(n_points, monotone = monotone)
c = exT3.cost_f(x, y, z)
ws = vmot.generate_working_sample_T3(u, v, w, x, y, z, c)


# D, H, c = vmot.mtg_dual_value(ws, model, d, T, monotone)
pi_hat, lbound, ubound = vmot.mtg_numeric_pi_hat(ws, model, d, T, monotone = monotone, opt_parameters = opt_parameters)
print('T', T)
print('monotone?', monotone)
print('margin', ubound)
print('pi_hat total mass', pi_hat.sum())

normalize = True
if normalize:
    pi_hat /= pi_hat.sum()
    
# set_T2_mono = [x.copy(), y.copy(), pi_hat.copy()]  20
# set_T2_full = [x.copy(), y.copy(), pi_hat.copy()]  17
# set_T3_mono = [x.copy(), y.copy(), z.copy(), pi_hat.copy()]  18
# set_T3_full = [x.copy(), y.copy(), z.copy(), pi_hat.copy()]  17
    
# T = 2
heatmap = pd.DataFrame({ 'x1'  : x[:,0],
                         'x2'  : x[:,1],
                         'y1'  : y[:,0],
                         'y2'  : y[:,1],
                         'p'   : pi_hat  })

scale = 17. * max(pi_hat)
heat = heatmap.iloc[pi_hat > scale * np.random.random(len(pi_hat))]
print(len(heat))
plot_heatmap_T2(heat, x, y)


# T = 3
heatmap = pd.DataFrame({ 'x1'  : x[:,0],
                         'x2'  : x[:,1],
                         'y1'  : y[:,0],
                         'y2'  : y[:,1],
                         'z1'  : z[:,0],
                         'z2'  : z[:,1],
                         'p'   : pi_hat  })

scale = 1. * max(pi_hat)
heat = heatmap.iloc[pi_hat > scale * np.random.random(len(pi_hat))]
print(len(heat))
plot_heatmap_T3(heat, x, y, z)






# plot of four last-period maps
s = 4
c = 'black'
a = .1
fig, ax = pl.subplots(1, 4, sharex=True, sharey=True, figsize=[13.9,4.1])

# T2, full
x, y, pi_hat = set_T2_full
heatmap = pd.DataFrame({ 'x1': x[:,0], 'x2': x[:,1], 'y1': y[:,0], 'y2': y[:,1], 'p': pi_hat  })
scale = 16.5 * max(pi_hat)
heat = heatmap.iloc[pi_hat > scale * np.random.random(len(pi_hat))]
print(len(heat))
ax[0].set_xlim([0.4, 1.1])
ax[0].add_patch(Rectangle((0.5, 0.0), 0.5, 1.5, ls=':', facecolor="none", ec='k'))
ax[0].set_title('T = 2, General')
ax[0].scatter(heat['y1'], heat['y2'], s=s, color=c, alpha=a)

# T2, mono
x, y, pi_hat = set_T2_mono
heatmap = pd.DataFrame({ 'x1': x[:,0], 'x2': x[:,1], 'y1': y[:,0], 'y2': y[:,1], 'p': pi_hat  })
scale = 21. * max(pi_hat)
heat = heatmap.iloc[pi_hat > scale * np.random.random(len(pi_hat))]
print(len(heat))
ax[1].set_xlim([0.4, 1.1])
ax[1].add_patch(Rectangle((0.5, 0.0), 0.5, 1.5, ls=':', facecolor="none", ec='k'))
ax[1].set_title('T = 2, Reduced')
ax[1].scatter(heat['y1'], heat['y2'], s=s, color=c, alpha=a)

# T3, full
x, y, z, pi_hat = set_T3_full
heatmap = pd.DataFrame({ 'x1': x[:,0], 'x2': x[:,1], 'y1': y[:,0], 'y2': y[:,1], 'z1': z[:,0], 'z2': z[:,1], 'p': pi_hat  })
scale = 17. * max(pi_hat)
heat = heatmap.iloc[pi_hat > scale * np.random.random(len(pi_hat))]
print(len(heat))
ax[2].set_xlim([0.4, 1.1])
ax[2].add_patch(Rectangle((0.5, 0.0), 0.5, 1.5, ls=':', facecolor="none", ec='k'))
ax[2].set_title('T = 3, General')
ax[2].scatter(heat['z1'], heat['z2'], s=s, color=c, alpha=a)

# T3, mono
x, y, z, pi_hat = set_T3_mono
heatmap = pd.DataFrame({ 'x1': x[:,0], 'x2': x[:,1], 'y1': y[:,0], 'y2': y[:,1], 'z1': z[:,0], 'z2': z[:,1], 'p': pi_hat  })
scale = 19. * max(pi_hat)
heat = heatmap.iloc[pi_hat > scale * np.random.random(len(pi_hat))]
print(len(heat))
ax[3].set_xlim([0.4, 1.1])
ax[3].add_patch(Rectangle((0.5, 0.0), 0.5, 1.5, ls=':', facecolor="none", ec='k'))
ax[3].set_title('T = 3, Reduced')
ax[3].scatter(heat['z1'], heat['z2'], s=s, color=c, alpha=a)




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

