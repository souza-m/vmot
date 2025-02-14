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
# import matplotlib.colors as mcolors
# from cycler import cycler


# example with cross-product cost and uniform marginals, d = 2, T = 3

# over all possible joint distributions of (X,Y)


# two marginals 1, 2
# three periods X, Y, Z
d = 2
T = 2

# inverse cumulative of U[0.5 , 1.0]
def F_inv_X1(u):
    return 0.5 + 0.5 * u

# inverse cumulative of U[0.5 , 1.0]
def F_inv_X2(u):
    return 0.5 + 0.5 * u

# inverse cumulative of U[0.5 , 1.0]
def F_inv_Y1(u):
    return 0.5 + 0.5 * u

# inverse cumulative of U[0.0 , 1.5]
def F_inv_Y2(u):
    return 1.5 * u


# generate synthetic sample
# u maps to x, v maps to y, w maps to z through the inverse cumulatives
def random_sample(n_points, monotone):
    u = np.random.random((n_points, 1 if monotone else 2))
    v = np.random.random((n_points, 2))
    
    if monotone:
        x = np.array([F_inv_X1(u[:,0]), F_inv_X2(u[:,0])]).T
    else:
        x = np.array([F_inv_X1(u[:,0]), F_inv_X2(u[:,1])]).T
    y = np.array([F_inv_Y1(v[:,0]), F_inv_Y2(v[:,1])]).T
                  
    return u, v, x, y
    

# cost function
def cost_f(x, y):
    return np.exp(y[:,0] + y[:,1])


# reference value
ref_value = 2.5562   # true VMOT solution, see Example 4.2 of REF

# u, v, x, y = random_sample(n_points=10000000, monotone = True)
# c = cost_f(x, y)
# sample_mean = c.mean()

# --- process batches and save models (takes long time) ---

# optimization parameters
opt_parameters = { 'gamma'           : 1000,    # penalization parameter
                   'epochs'          : 1,       # iteration parameter
                   'batch_size'      : 1000  }  # iteration parameter  

# batch control
I = 100            # total desired iterations
existing_i = 0     # last iteration saved
n_points = 100000   # sample points at each iteration

if existing_i == 0:
    # new random sample
    print('\niteration 1 (new model)\n')
    
    # regular coupling
    u, v, x, y = random_sample(n_points, monotone = False)
    c = cost_f(x, y)
    ws1 = vmot.generate_working_sample_T2(u, v, x, y, c)
    
    # monotone coupling
    u, v, x, y = random_sample(n_points, monotone = True)
    c = cost_f(x, y)
    ws2 = vmot.generate_working_sample_T2(u, v, x, y, c)
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
    
# load existing model
print(f'\nloading model {existing_i}')
model1, opt1, D1_series = vmot.load_results(f'uniform_full_d{d}_T{T}_{existing_i}')
model2, opt2, D2_series = vmot.load_results(f'uniform_mono_d{d}_T{T}_{existing_i}')

# iterative parsing
while existing_i < I:
    
    # new random sample
    print(f'\niteration {existing_i+1}\n')
    
    # regular coupling
    u, v, x, y = random_sample(n_points, monotone = False)
    c = cost_f(x, y)
    ws1 = vmot.generate_working_sample_T2(u, v, x, y, c)
    
    # monotone coupling
    u, v, x, y = random_sample(n_points, monotone = True)
    c = cost_f(x, y)
    ws2 = vmot.generate_working_sample_T2(u, v, x, y, c)
    print('samples generated, shapes ', ws1.shape, 'and', ws2.shape)
    
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
# vmot.dump_results([model1, D1_series], f'uniform_full_d{d}_T{T}_{existing_i}')
# vmot.dump_results([model2, D2_series], f'uniform_mono_d{d}_T{T}_{existing_i}')
model1, D1_series = vmot.load_results(f'uniform_full_d{d}_T{T}_{existing_i}')
model2, D2_series = vmot.load_results(f'uniform_mono_d{d}_T{T}_{existing_i}')


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
u, v, x, y = random_sample(n_points, monotone = monotone)
c = cost_f(x, y)
ws = vmot.generate_working_sample_T2(u, v, x, y, c)
model = model1
D, H, c, w = vmot.mtg_dual_value(ws, model, d, T, monotone)
pi_hat, lbound, ubound = vmot.mtg_numeric_pi_hat(ws, model, d, T, monotone = False, opt_parameters = opt_parameters)

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

