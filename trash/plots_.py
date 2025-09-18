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


# density map

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



folder='gamma_1000/'
existing_i = 150
tail_cut = 100
tail_size = existing_i - tail_cut
opt_parameters = { 'gamma'           : 1000,    # penalization parameter
                   'epochs'          : 1,       # iteration parameter
                   'batch_size'      : 1000  }  # iteration parameter  
OT_value = (np.e ** 0.5) * (np.e ** 2 - 1.) / 2   # analytical true value Ex. 4.2
VMOT_value = 5.1124   # analytical true value Ex. 4.2
d = 2
n_points = 1000000


# print statistics and load batch data for later plotting
value_series = []
density_set = []
for T in [2, 3]:
    for monotone in [False, True]:
        
        title = f'Numerical value (T = {T}, ' + ('reduced' if monotone else 'general') + ')'
        print()
        print(title)

        # historical series
        model, D_series, H_series, P_series = vmot.load_results(folder=folder, label=f'uniform_d{d}_T{T}_{existing_i}_{"mono" if monotone else "full"}_cpu')
        evo = np.array(D_series)
        print(f'full series   mean = {evo.mean():8.5f},   std = {evo.std():8.5f}')
        print(f'tail series   mean = {evo[tail_cut:].mean():8.5f},   std = {evo[tail_cut:].std():8.5f}')
        value_series.append(evo)
        
        if T == 2:
            u, v, x, y = exT2.random_sample(n_points, monotone = monotone)
            c = exT2.cost_f(x, y)
            ws = vmot.generate_working_sample_T2(u, v, x, y, c)
            
            pi_hat, lmargin, umargin = vmot.mtg_numeric_pi_hat(ws, model, d, T, monotone = monotone, opt_parameters = opt_parameters)
            print('upper margin', umargin)
            print('pi_hat total mass', pi_hat.sum())
            
            normalize = True
            if normalize:
                pi_hat /= pi_hat.sum()
                
            # T = 2
            heatmap = pd.DataFrame({ 'x1'  : x[:,0],
                                     'x2'  : x[:,1],
                                     'y1'  : y[:,0],
                                     'y2'  : y[:,1],
                                     'p'   : pi_hat  })
        
        else:
            u, v, w, x, y, z = exT3.random_sample(n_points, monotone = monotone)
            c = exT3.cost_f(x, y, z)
            ws = vmot.generate_working_sample_T3(u, v, w, x, y, z, c)
                
            pi_hat, lmargin, umargin = vmot.mtg_numeric_pi_hat(ws, model, d, T, monotone = monotone, opt_parameters = opt_parameters)
            print(pi_hat.sum())
            
        
            # T = 3
            heatmap = pd.DataFrame({ 'x1'  : x[:,0],
                                     'x2'  : x[:,1],
                                     'y1'  : y[:,0],
                                     'y2'  : y[:,1],
                                     'z1'  : z[:,0],
                                     'z2'  : z[:,1],
                                     'p'   : pi_hat  })
            
        pi_hat /= pi_hat.sum()
        cut = np.quantile(pi_hat, 1 - (5000 / n_points))
        # heat = heatmap.iloc[pi_hat > cut]
        heat = heatmap.iloc[pi_hat * np.random.random(len(pi_hat)) > cut / 2]
        density_set.append([x.copy(), y.copy(), pi_hat.copy(), umargin, cut])
        
        if T == 2:
            plot_heatmap_T2(heat, x, y)
            
        if T == 3:
            plot_heatmap_T3(heat, x, y, z)
        

# evolution of numeric values
labels = ['Full dimension', 'Reduced dimension', 'OT value', 'VMOT value (true)']
title_T2 = 'Numerical value (T = 2)'
title_T3 = 'Numerical value (T = 3)'
fig, ax = pl.subplots(1, 2, sharex=True, sharey=True, figsize=[9,4])
size = len(value_series[0])

_range = range(tail_cut + 1, size + 1)


tolerance_values = [
    (0.01, 0.05),  # First line
    (0.02, 0.03),  # Second line
    (0.015, 0.04),  # Third line
    (0.025, 0.02)   # Fourth line
]

# T = 2
# Plot first two series with tolerance bands
line_handles = []
for i in range(2):
    values = value_series[i][tail_cut:]
    upper_tol = values + tolerance_values[i][0]
    lower_tol = values - tolerance_values[i][1]

    line, = ax[0].plot(_range, values)  # Plot the main line
    ax[0].fill_between(_range, lower_tol, upper_tol, color=line.get_color(), alpha=0.2)  # Tolerance band
    line_handles.append(line)

# ax[0].plot(_range, value_series[0][tail_cut:])
# ax[0].plot(_range, value_series[1][tail_cut:])
line_handles.append(ax[0].axhline(OT_value, linestyle='--', linewidth=2, color='black'))
line_handles.append(ax[0].axhline(VMOT_value, linestyle='--', linewidth=2, color='red'))
ax[0].legend(handles=line_handles, loc='center right', labels=labels)
ax[0].scatter(_range, value_series[0][tail_cut:], s=5)
ax[0].scatter(_range, value_series[1][tail_cut:], s=5)
ax[0].set_xlabel('Iteration')
ax[0].set_title(title_T2)

# T = 3
# Plot second two series with tolerance bands
line_handles = []
for i in range(2, 4):
    values = value_series[i][tail_cut:]
    upper_tol = values + tolerance_values[i][0]
    lower_tol = values - tolerance_values[i][1]

    line, = ax[1].plot(_range, values)  # Plot the main line
    ax[1].fill_between(_range, lower_tol, upper_tol, color=line.get_color(), alpha=0.2)  # Tolerance band
    line_handles.append(line)

# ax[1].plot(_range, value_series[2][tail_cut:])
# ax[1].plot(_range, value_series[3][tail_cut:])
line_handles.append(ax[1].axhline(OT_value, linestyle='--', linewidth=2, color='black'))
ax[1].legend(handles=line_handles, labels=labels[:-1], loc='best')
ax[1].scatter(_range, value_series[2][tail_cut:], s=5)
ax[1].scatter(_range, value_series[3][tail_cut:], s=5)
ax[1].set_xlabel('Iteration')
ax[1].set_title(title_T3)

fig.show()




# margins based on beta_conjugate

# T = 2
monotone = True   # choose
model = model2 if monotone else model1
u, v, x, y = exT2.random_sample(n_points, monotone = monotone)
c = exT2.cost_f(x, y)
ws = vmot.generate_working_sample_T2(u, v, x, y, c)

# T = 3
monotone = True   # choose
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
    
# set_T2_full = [x.copy(), y.copy(), pi_hat.copy()]  1
# set_T2_mono = [x.copy(), y.copy(), pi_hat.copy()]  1.7
# set_T3_full = [x.copy(), y.copy(), z.copy(), pi_hat.copy()]  .75
# set_T3_mono = [x.copy(), y.copy(), z.copy(), pi_hat.copy()]  .95
    
# T = 2
heatmap = pd.DataFrame({ 'x1'  : x[:,0],
                         'x2'  : x[:,1],
                         'y1'  : y[:,0],
                         'y2'  : y[:,1],
                         'p'   : pi_hat  })

scale = 1.7 * max(pi_hat)
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

scale = .95 * max(pi_hat)
heat = heatmap.iloc[pi_hat > scale * np.random.random(len(pi_hat))]
print(len(heat))
plot_heatmap_T3(heat, x, y, z)






# plot of four last-period maps

s = 4
c = 'black'
a = .1
fig, ax = pl.subplots(1, 4, sharex=True, sharey=True, figsize=[13.9,4.1])

# T2, full
x, y, pi_hat, umargin, cut = density_set[0]
heatmap = pd.DataFrame({ 'x1': x[:,0], 'x2': x[:,1], 'y1': y[:,0], 'y2': y[:,1], 'p': pi_hat  })
scale = 1. * max(pi_hat)
heat = heatmap.iloc[pi_hat > scale * np.random.random(len(pi_hat))]
print(len(heat))
ax[0].set_xlim([0.4, 1.1])
ax[0].add_patch(Rectangle((0.5, 0.0), 0.5, 1.5, ls=':', facecolor="none", ec='k'))
ax[0].set_title('T = 2, General')
ax[0].scatter(heat['y1'], heat['y2'], s=s, color=c, alpha=a)

# T2, mono
x, y, pi_hat, umargin, cut = density_set[1]
heatmap = pd.DataFrame({ 'x1': x[:,0], 'x2': x[:,1], 'y1': y[:,0], 'y2': y[:,1], 'p': pi_hat  })
scale = 1.7 * max(pi_hat)
heat = heatmap.iloc[pi_hat > scale * np.random.random(len(pi_hat))]
print(len(heat))
ax[1].set_xlim([0.4, 1.1])
ax[1].add_patch(Rectangle((0.5, 0.0), 0.5, 1.5, ls=':', facecolor="none", ec='k'))
ax[1].set_title('T = 2, Reduced')
ax[1].scatter(heat['y1'], heat['y2'], s=s, color=c, alpha=a)

# T3, full
x, y, z, pi_hat, umargin, cut = density_set[2]
heatmap = pd.DataFrame({ 'x1': x[:,0], 'x2': x[:,1], 'y1': y[:,0], 'y2': y[:,1], 'z1': z[:,0], 'z2': z[:,1], 'p': pi_hat  })
scale = .75 * max(pi_hat)
heat = heatmap.iloc[pi_hat > scale * np.random.random(len(pi_hat))]
print(len(heat))
ax[2].set_xlim([0.4, 1.1])
ax[2].add_patch(Rectangle((0.5, 0.0), 0.5, 1.5, ls=':', facecolor="none", ec='k'))
ax[2].set_title('T = 3, General')
ax[2].scatter(heat['z1'], heat['z2'], s=s, color=c, alpha=a)

# T3, mono
x, y, z, pi_hat, umargin, cut = density_set[3]
heatmap = pd.DataFrame({ 'x1': x[:,0], 'x2': x[:,1], 'y1': y[:,0], 'y2': y[:,1], 'z1': z[:,0], 'z2': z[:,1], 'p': pi_hat  })
scale = .95 * max(pi_hat)
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

