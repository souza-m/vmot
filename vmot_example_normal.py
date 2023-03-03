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
n = 40   # marginal sample/grid size
full_size = n ** (2 * d)
print(f'full size: {full_size}')
opt_parameters = { 'penalization'    : 'L2',
                   'beta_multiplier' : 1,
                   'gamma'           : 1000,
                   'batch_size'      : n ** d,   # no special formula for this, using sqrt of working sample size
                   'macro_epochs'    : 3,
                   'micro_epochs'    : 10      }


# 0. marginals sampling example

def generate_xi(n_points, i, clip_normal = None):
    if clip_normal is None:   # noisy tails included
        return np.random.normal(loc=0.0, scale = x_normal_scale[i], size=n_points)
    else:
        # clip tails
        rv = stats.truncnorm(-clip_normal, clip_normal)
        z = rv.rvs(size = n_points)
        return z * x_normal_scale[i]

def generate_yi(n_points, i, clip_normal = None):
    if clip_normal is None:   # noisy tails included
        return np.random.normal(loc=0.0, scale = y_normal_scale[i], size=n_points)
    else:
        # clip tails
        rv = stats.truncnorm(-clip_normal, clip_normal)
        z = rv.rvs(size = n_points)
        return z * y_normal_scale[i]

np.random.seed(0)
xi_list = [generate_xi(n, i) for i in range(d)]
yi_list = [generate_yi(n, i) for i in range(d)]
xy_set0 = vmot.xi_yi_to_xy_set(xi_list, yi_list, monotone_x = False)
ws0 = vmot.generate_working_sample(xy_set0, cost_f, uniform_theta = True)


# 1. full sampling example

def generate_xy_set(n_points, clip_normal = None):
    if clip_normal is None:   # noisy tails included
        random_sample = np.random.normal(loc=0.0, scale=1, size = [n_points, 4])
        # random_sample = np.array([np.random.normal(loc=0.0, scale=1, size = n_points) for i in range(4)]).T
        xy = random_sample * (x_normal_scale + y_normal_scale)
        return xy
    else:   # clip tails
        print('not implemented')
        return None

np.random.seed(0)
xy_set1 = generate_xy_set(n_points = full_size)
ws1 = vmot.generate_working_sample(xy_set1, cost_f, uniform_theta = True)


# 2. quantile grid example

def inv_cum_x(q, i):
    return norm.ppf(q) * x_normal_scale[i]

def inv_cum_y(q, i):
    return norm.ppf(q) * y_normal_scale[i]

def cum_x(q, i):
    return norm.cdf(q / x_normal_scale[i])

def cum_y(q, i):
    return norm.cdf(q / y_normal_scale[i])

q_set2 = vmot.grid_to_q_set(n, d, monotone_x = False)
ws2 = vmot.generate_working_sample_q(q_set2, inv_cum_x, inv_cum_y, cost_f, uniform_theta = True)


# 3. quantile set example

def generate_q_set(n_points):
    uniform_sample = np.random.random((n_points, 2*d))
    return uniform_sample

def xy_set_to_q_set(xy_set):
    d = int(xy_set.shape[1] / 2)
    x = xy_set[:, :d]
    y = xy_set[:, d:]
    qx = np.array([cum_x(x[:,i], i) for i in range(d)]).T
    qy = np.array([cum_y(y[:,i], i) for i in range(d)]).T
    q_set = np.hstack([qx, qy])
    return q_set

def q_set_to_sample(q_set):
    d = int(q_set.shape[1] / 2)
    qx = q_set[:, :d]
    qy = q_set[:, d:]
    x = np.array([inv_cum_x(qx[:,i], i) for i in range(d)]).T
    y = np.array([inv_cum_y(qy[:,i], i) for i in range(d)]).T
    xy_set = np.hstack([x, y])
    return xy_set

n_points = full_size
np.random.seed(0)
# q_set = generate_q_set(n_points)   # new random independent sample
q_set3 = xy_set_to_q_set(xy_set1)    # same sample as in number 1
xy_set3 = q_set_to_sample(q_set3)   # consistency check
assert np.isclose(xy_set1, xy_set3).all()

ws3 = vmot.generate_working_sample_q(q_set3, inv_cum_x, inv_cum_y, cost_f, uniform_theta = True)


vmot.plot_sample_2d(xy_set0, 'q_set sampling')
vmot.plot_sample_2d(xy_set1, 'q_set sampling')
vmot.plot_sample_2d(q_set2, 'q_set sampling')
vmot.plot_sample_2d(q_set3, 'q_set sampling')



model0, D_evo0, s_evo0, H_evo0, P_evo0 = vmot.mtg_train(ws0, opt_parameters, verbose = 100)
model1, D_evo1, s_evo1, H_evo1, P_evo1 = vmot.mtg_train(ws1, opt_parameters, verbose = 100)
model2, D_evo2, s_evo2, H_evo2, P_evo2 = vmot.mtg_train(ws2, opt_parameters, verbose = 100)
model3, D_evo3, s_evo3, H_evo3, P_evo3 = vmot.mtg_train(ws3, opt_parameters, verbose = 100)


# convergence plots
band_size = 1
# labels = ['sample marginals', 'sample set', 'quantile grid', 'quantile set']
# D_evo_list = [D_evo0, D_evo1, D_evo2, D_evo3]
# H_evo_list = [H_evo0, H_evo1, H_evo2, H_evo3]
# s_evo_list = [s_evo0, s_evo1, s_evo2, s_evo3]
labels = ['sample set', 'quantile grid', 'quantile set']
D_evo_list = [D_evo1, D_evo2, D_evo3]
H_evo_list = [H_evo1, H_evo2, H_evo3]
s_evo_list = [s_evo1, s_evo2, s_evo3]

# convergence comparison (D)
pl.figure(figsize = [12,12])   # plot in two iterations to have a clean legend
[pl.plot(D_evo) for D_evo in D_evo_list]
pl.legend(labels)
for D_evo, s_evo in zip(D_evo_list, s_evo_list):
    pl.fill_between(range(len(D_evo)),
                    np.array(D_evo) + np.array(band_size * s_evo),
                    np.array(D_evo) - np.array(band_size * s_evo), alpha = .3, facecolor = 'grey')
pl.axhline(ref_value, linestyle=':', color='black')

# convergence comparison (D + H)
pl.figure(figsize = [12,12])   # plot in two iterations to have a clean legend
[pl.plot(np.array(D_evo) + np.array(H_evo)) for D_evo, H_evo in zip(D_evo_list, H_evo_list)]
pl.legend(labels)
for D_evo, H_evo, s_evo in zip(D_evo_list, H_evo_list, s_evo_list):
    pl.fill_between(range(len(D_evo)),
                    np.array(D_evo) + np.array(H_evo) + band_size * np.array(s_evo),
                    np.array(D_evo) + np.array(H_evo) - band_size * np.array(s_evo), alpha = .3, facecolor = 'grey')
pl.axhline(ref_value, linestyle=':', color='black')




# tests
# vmot.plot_sample_2d(working_sample, 'q_set sampling')
# D_mean, H_mean, pi_star = vmot.dual_value(working_sample, opt_parameters, model4)

# symmetrical quantiles (range [-1/2, 1/2])
# working_sample_ = vmot.generate_working_sample_q_set(q_set, inv_cum_x, inv_cum_y, cost_f,
#                                                      symmetrical = True,
#                                                      monotone_x = False,
#                                                      uniform_theta = True)
# model5, D_series5, s_series5, H_series5, P_series5 = vmot.mtg_train(working_sample_, opt_parameters, verbose = 100)

# previous, range [-1, 1]
# model5a, D_series5a, s_series5a, H_series5a, P_series5a = model5.copy(), D_series5.copy(), s_series5.copy(), H_series5.copy(), P_series5.copy()

# previous, range [-1000, 1000]
# model5b, D_series5b, s_series5b, H_series5b, P_series5b = model5.copy(), D_series5.copy(), s_series5.copy(), H_series5.copy(), P_series5.copy()


