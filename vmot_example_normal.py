# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:35:25 2021
@author: souzam
PyTorch implementation of Eckstein and Kupper 2019 - Computation of Optimal Transport...
"""


import numpy as np
import matplotlib.pyplot as pl
from scipy.stats import norm
import scipy.stats as stats

import vmot_core as vmot


# choose d and marginal distributions
d = 2
x_normal_scale = [1.0, 1.0]
y_normal_scale = [1.5, 2.0]

# cost function
A = 0
B = 1
def cost_f(x, y):
    # cost = A.x1.x2 + B.y1.y2
    return A * x[:,0] * x[:,1] + B * y[:,0] * y[:,1]

# reference value (Prop. 3.7)
sig1 = x_normal_scale[0]
sig2 = x_normal_scale[1]
rho1 = y_normal_scale[0]
rho2 = y_normal_scale[1]
lam1 = np.sqrt(rho1 ** 2 - sig1 ** 2)
lam2 = np.sqrt(rho2 ** 2 - sig2 ** 2)
ref_value = (A + B) * sig1 * sig2 + B * lam1 * lam2

# common parameters
n = 30   # marginal sample/grid size
full_size = n ** (2 * d)
print(f'full size: {full_size}')
opt_parameters = { 'penalization' : 'L2',
                   'b_multiplier' : .1,
                   'gamma'        : 1000,
                   'batch_size'   : n ** d,   # no special formula for this, using sqrt of working sample size
                   'macro_epochs' : 5,
                   'micro_epochs' : 5      }


# 1. marginals sampling example

def generate_x_sample(n_points, i, clip_normal = None):
    if clip_normal is None:   # noisy tails included
        return np.random.normal(loc=0.0, scale = x_normal_scale[i], size=n_points)
    else:
        # clip tails
        rv = stats.truncnorm(-clip_normal, clip_normal)
        z = rv.rvs(size = n_points)
        return z * x_normal_scale[i]

def generate_y_sample(n_points, i, clip_normal = None):
    if clip_normal is None:   # noisy tails included
        return np.random.normal(loc=0.0, scale = y_normal_scale[i], size=n_points)
    else:
        # clip tails
        rv = stats.truncnorm(-clip_normal, clip_normal)
        z = rv.rvs(size = n_points)
        return z * y_normal_scale[i]

np.random.seed(0)
xi_sample_list = [generate_x_sample(n, i) for i in range(d)]
yi_sample_list = [generate_y_sample(n, i) for i in range(d)]
working_sample = vmot.generate_working_sample_marginals(xi_sample_list, yi_sample_list, cost_f,
                                                        monotone_x = False,
                                                        uniform_theta = True)
vmot.plot_sample_2d(working_sample, 'marginals sampling')
# model1, D_series1, s_series1, H_series1, P_series1 = vmot.mtg_train(working_sample, opt_parameters, verbose = 1000)
# D_mean, H_mean, pi_star = vmot.dual_value(working_sample, opt_parameters, model)

# 2. full sampling example

def generate_xy_sample(n_points, clip_normal = None):   # coupling in ['independent', 'positive', 'negative', 'straight']
    if clip_normal is None:   # noisy tails included
        random_sample = np.random.normal(loc=0.0, scale=x_normal_scale[0], size = [n_points, 4])
    else:   # clip tails
        rv = stats.truncnorm(-clip_normal, clip_normal)
        random_sample = rv.rvs(size = [n_points, 4])
    xy = random_sample * (x_normal_scale + y_normal_scale)
    return xy

n_points = full_size
np.random.seed(0)
xy_sample = generate_xy_sample(n_points)
working_sample = vmot.generate_working_sample(xy_sample, cost_f,
                                              uniform_theta = True)
vmot.plot_sample_2d(working_sample, 'x-y sampling')
model2, D_series2, s_series2, H_series2, P_series2 = vmot.mtg_train(working_sample, opt_parameters, verbose = 1000)
D_mean, H_mean, pi_star = vmot.dual_value(working_sample, opt_parameters, model2)


# 3. quantile example

def inv_cum_x(xhat, i):
    return norm.ppf(xhat) * x_normal_scale[i]

def inv_cum_y(xhat, i):
    return norm.ppf(xhat) * y_normal_scale[i]

working_sample = vmot.generate_working_sample_quantile(n, d, inv_cum_x, inv_cum_y, cost_f,
                                                       monotone_x = False,
                                                       uniform_theta = True)
vmot.plot_sample_2d(working_sample, 'quantile sampling')
model3, D_series3, s_series3, H_series3, P_series3 = vmot.mtg_train(working_sample, opt_parameters, verbose = 1000)
D_mean, H_mean, pi_star = vmot.dual_value(working_sample, opt_parameters, model3)



# convergence comparison (D)
band_size = 1
labels = ['sample', 'quantile']
pl.figure(figsize = [12,12])   # plot in two iterations to have a clean legend
# D_series_list = [D_series1, D_series2, D_series3]
# s_series_list = [s_series1, s_series2, s_series3]
D_series_list = [D_series2, D_series3]
s_series_list = [s_series2, s_series3]
[pl.plot(D_series) for D_series in D_series_list]
pl.legend(labels)
for D_series, s_series in zip(D_series_list, s_series_list):
    pl.fill_between(range(len(D_series)),
                    np.array(D_series) + np.array(band_size * s_series),
                    np.array(D_series) - np.array(band_size * s_series), alpha = .3, facecolor = 'grey')
pl.axhline(ref_value, linestyle=':', color='black')

# convergence comparison (D + H)
band_size = 1
pl.figure(figsize = [12,12])   # plot in two iterations to have a clean legend
# D_series_list = [D_series1, D_series2, D_series3]
# s_series_list = [s_series1, s_series2, s_series3]
D_series_list = [D_series2, D_series3]
H_series_list = [H_series2, H_series3]
s_series_list = [s_series2, s_series3]
[pl.plot(np.array(D_series) + np.array(H_series)) for D_series, H_series in zip(D_series_list, H_series_list)]
pl.legend(labels)
pl.legend(list(range(len(D_series_list))))
for D_series, H_series, s_series in zip(D_series_list, H_series_list, s_series_list):
    pl.fill_between(range(len(D_series)),
                    np.array(D_series) + np.array(H_series) + band_size * np.array(s_series),
                    np.array(D_series) + np.array(H_series) - band_size * np.array(s_series), alpha = .3, facecolor = 'grey')
pl.axhline(ref_value, linestyle=':', color='black')
