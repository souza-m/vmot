# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:35:25 2021
@author: souzam
PyTorch implementation of Eckstein and Kupper 2019 - Computation of Optimal Transport...
"""

# Example 1: solve cross-product cost with normal marginals, d = 2
#
#   Cost function:   cost_f(x, y) = y1 * y2
#
#   Primal:          max cost
#            equiv.  min C := minus_cost
#
#   Dual:            max D := sum{phi} + sum{psi} + sum{h.L}
#                    st  D <= minus_cost
#
#   Penalized dual:  min (-D) + b(D - minus_cost)
#
# Coupling structure:
#   (1)   independent (4 dimensions))
#   (2)   montone, dimension reduction on x (3 dimensions)
#
# Method: use vmot_core to generate "working sample" objects to be used in the optimization loop.
# The optimization loop performs the dual approximation based on E&K21.
# We experiment with dimensionlality reduction and random vs grid sampling methods.


import numpy as np
import matplotlib.pyplot as pl
from scipy.stats import norm
import pickle

import vmot


# choose d and marginal distributions
d = 2
x_normal_scale = [1.0, 1.0]
y_normal_scale = [1.5, 2.0]

# cost function to be minimized
A = 0
B = 1
def cost_f(x, y):
    # cost = A.x1.x2 + B.y1.y2
    return A * x[:,0] * x[:,1] + B * y[:,0] * y[:,1]

# to be maximized
def minus_cost_f(x, y):
    return -cost_f(x, y)

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
# mono_size = n ** (d + 1)
mono_size = full_size
print(f'full size: {full_size}')
print(f'mono size: {mono_size}')


# 1. independent

def generate_qset(n_points, d):
    uniform_sample = np.random.random((n_points, 2*d))
    return uniform_sample

def inv_cum_xi(q, i):
    return norm.ppf(q) * x_normal_scale[i]

def inv_cum_yi(q, i):
    return norm.ppf(q) * y_normal_scale[i]

def cum_x(q, i):
    return norm.cdf(q / x_normal_scale[i])

def cum_y(q, i):
    return norm.cdf(q / y_normal_scale[i])

n_points = full_size
np.random.seed(0)
qset1 = generate_qset(n_points, d)
ws1 = vmot.generate_working_sample_uv(qset1, inv_cum_xi, inv_cum_yi, minus_cost_f, uniform_weight = True)

print('independent coupling sample shape ', qset1.shape)


# 2. monotone

def generate_qset_mono(n_points, d):
    uniform_sample = np.random.random((n_points, d+1))
    return uniform_sample

def inv_cum_x(q):
    z = norm.ppf(q)
    return np.array([z * x_normal_scale[i] for i in range(d)]).T

n_points = mono_size
np.random.seed(123)
qset2 = generate_qset_mono(n_points, d)
ws2 = vmot.generate_working_sample_uv_mono(qset2, inv_cum_x, inv_cum_yi, minus_cost_f, uniform_weight = True)

print('monotone coupling sample shape ', qset2.shape)


# check
# qset1
# ws1[:,:4]

# q_set = qset1
# qx = q_set[:, :d]
# qy = q_set[:, d:]
# x = np.array([inv_cum_xi(qx[:,i], i) for i in range(d)]).T
# y = np.array([inv_cum_yi(qy[:,i], i) for i in range(d)]).T
# x
# y
# ws1[:,[4,5]]
    
# qset2
# ws2[:,:3]

# q_set = qset2
# qx = q_set[:, 0]        # n x 1
# qy = q_set[:, 1:d+1]    # n x d
# x = inv_cum_x(qx)       # n x d
# y = np.array([inv_cum_yi(qy[:,i], i) for i in range(d)]).T
# x
# y
# ws2[:,[3,4]]

# pl.figure()
# pl.plot(ws1[:,6])
# pl.plot(ws2[:,5])

# train

opt_parameters = { 'penalization'    : 'L2',
                   'beta_multiplier' : 10,
                   'gamma'           : 1000,
                   'batch_size'      : n ** d,   # no special formula for this, using sqrt of working sample size
                   'macro_epochs'    : 2,
                   'micro_epochs'    : 10      }

model1, D_evo1, H_evo1, P_evo1, ds_evo1, hs_evo1 = vmot.mtg_train(ws1, opt_parameters, monotone = False, verbose = 100)

model2, D_evo2, H_evo2, P_evo2, ds_evo2, hs_evo2 = vmot.mtg_train(ws2, opt_parameters, monotone = True, verbose = 100)


# test

D1, H1, pi_star1 = vmot.mtg_dual_value(model1, ws1, opt_parameters)
D2, H2, pi_star2 = vmot.mtg_dual_value(model2, ws2, opt_parameters)

sample = torch.tensor(ws1[:5]).float()
model = model1
phi1, psi1, h1, L1, c1, pi1 = mtg_parse(model, sample)

sample = torch.tensor(ws2[:5]).float()
model = model2
phi2, psi2, h2, L2, c2, pi2 = mtg_parse(model, sample)

def convergence_plot(value_series_list, labels, ref_value = None):
    pl.figure(figsize = [12,12])   # plot in two iterations to have a clean legend
    for v in value_series_list:
        pl.plot(v)
    pl.legend(labels)
    if not ref_value is None:
        pl.axhline(ref_value, linestyle=':', color='black')
    pl.show()


evo1 = -np.array(D_evo1)
evo2 = -np.array(D_evo2)
# convergence_plot([evo1], ['independent'], ref_value)
convergence_plot([evo1, evo2], ['independent', 'monotone'], ref_value)



# _D_evo1, _s_evo1, _H_evo1, _P_evo1 = D_evo1.copy(), s_evo1.copy(), H_evo1.copy(), P_evo1.copy()
# _D_evo2, _s_evo2, _H_evo2, _P_evo2 = D_evo2.copy(), s_evo2.copy(), H_evo2.copy(), P_evo2.copy()

model1, D_evo1, s_evo1, H_evo1, P_evo1 = vmot.mtg_train(ws1,
                   { 'penalization'    : 'L2',
                     'beta_multiplier' : 10,
                     'gamma'           : 1000,
                     'batch_size'      : n ** d,
                     'macro_epochs'    : 2,
                     'micro_epochs'    : 20      }, model = model1, monotone = False, verbose = 100)

model2, D_evo2, s_evo2, H_evo2, P_evo2 = vmot.mtg_train(ws2,
                   { 'penalization'    : 'L2',
                     'beta_multiplier' : 10,
                     'gamma'           : 1000,
                     'batch_size'      : n ** d,
                     'macro_epochs'    : 2,
                     'micro_epochs'    : 20      }, model = model2, monotone = True, verbose = 100)





evo1a = -np.array(D_evo1a)
evo2a = -np.array(D_evo2a)
# convergence_plot([evo1], ['independent'], ref_value)
convergence_plot([evo1a, evo2a], ['independent', 'monotone'], ref_value)






results = [model1, D_evo1, s_evo1, H_evo1, P_evo1, model2, D_evo2, s_evo2, H_evo2, P_evo2]

# dump
_dir = './model_dump/'
_file = 'quantile_dim_results.pickle'
_path = _dir + _file
with open(_path, 'wb') as file:
    pickle.dump(results, file)
print('model saved to ' + _path)











xy_set1 = qset1[:,:4]
xy_set2 = np.hstack([np.tile(qset2[:,0], (2, 1)).T, qset2[:,1:3]])
vmot.plot_sample_2d(xy_set1, 'q_set sampling')
vmot.plot_sample_2d(xy_set2, 'q_set sampling')














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
ws1 = vmot.generate_working_sample(xy_set1, minus_cost_f, uniform_theta = True)


# 2. quantile grid example

q_set2 = vmot.grid_to_q_set(n, d, monotone_x = False)
ws2 = vmot.generate_working_sample_q(q_set2, inv_cum_x, inv_cum_y, minus_cost_f, uniform_theta = True)




# 3. quantile set example



# 4. grid on original domain

xy_set4 = q_set_to_sample(q_set2)
ws4 = vmot.generate_working_sample(xy_set4, minus_cost_f, uniform_theta = True)


# vmot.plot_sample_2d(xy_set0, 'q_set sampling')
# vmot.plot_sample_2d(xy_set1, 'q_set sampling')
# vmot.plot_sample_2d(q_set2, 'q_set sampling')
# vmot.plot_sample_2d(q_set3, 'q_set sampling')
# vmot.plot_sample_2d(xy_set4, 'q_set sampling')



# beta_multiplier = 10
# model0, D_evo0, s_evo0, H_evo0, P_evo0 = vmot.mtg_train(ws0, opt_parameters, verbose = 100)
# model1, D_evo1, s_evo1, H_evo1, P_evo1 = vmot.mtg_train(ws1, opt_parameters, verbose = 100)
# model2, D_evo2, s_evo2, H_evo2, P_evo2 = vmot.mtg_train(ws2, opt_parameters, verbose = 100)
# model3, D_evo3, s_evo3, H_evo3, P_evo3 = vmot.mtg_train(ws3, opt_parameters, verbose = 100)
# model4, D_evo4, s_evo4, H_evo4, P_evo4 = vmot.mtg_train(ws4, opt_parameters, verbose = 100)

# D0, s0, H0, theta_star0 = vmot.mtg_dual_value(model0, ws0, opt_parameters)
# D1, s1, H1, theta_star1 = vmot.mtg_dual_value(model1, ws1, opt_parameters)
# D2, s2, H2, theta_star2 = vmot.mtg_dual_value(model2, ws2, opt_parameters)
# D3, s3, H3, theta_star3 = vmot.mtg_dual_value(model3, ws3, opt_parameters)
# D4, s4, H4, theta_star4 = vmot.mtg_dual_value(model4, ws4, opt_parameters)

# # outputs
# print(f'target {ref_value:8.4f}    D0 {-D0:8.4f}    D1 {-D1:8.4f}    D2 {-D2:8.4f}    D3 {-D3:8.4f}')

# theta_star0 = theta_star0 / theta_star0.sum()
# theta_star1 = theta_star1 / theta_star1.sum()
# theta_star2 = theta_star2 / theta_star2.sum()
# theta_star3 = theta_star3 / theta_star3.sum()
# theta_star4 = theta_star4 / theta_star4.sum()
# theta_star0 = 0.5 * theta_star0 + 0.5 * np.ones(len(theta_star0)) / len(theta_star0)
# theta_star1 = 0.5 * theta_star1 + 0.5 * np.ones(len(theta_star1)) / len(theta_star1)
# theta_star2 = 0.5 * theta_star2 + 0.5 * np.ones(len(theta_star2)) / len(theta_star2)
# theta_star3 = 0.5 * theta_star3 + 0.5 * np.ones(len(theta_star3)) / len(theta_star3)
# theta_star4 = 0.5 * theta_star4 + 0.5 * np.ones(len(theta_star4)) / len(theta_star4)

# # new round with same old weights for comparison
# _model0, _D_evo0, _s_evo0, _H_evo0, _P_evo0 = vmot.mtg_train(ws0, opt_parameters, model = model0, verbose = 100)
# _model1, _D_evo1, _s_evo1, _H_evo1, _P_evo1 = vmot.mtg_train(ws1, opt_parameters, model = model1, verbose = 100)
# _model2, _D_evo2, _s_evo2, _H_evo2, _P_evo2 = vmot.mtg_train(ws2, opt_parameters, model = model2, verbose = 100)
# _model3, _D_evo3, _s_evo3, _H_evo3, _P_evo3 = vmot.mtg_train(ws3, opt_parameters, model = model3, verbose = 100)
# _model4, _D_evo4, _s_evo4, _H_evo4, _P_evo4 = vmot.mtg_train(ws4, opt_parameters, model = model4, verbose = 100)

# # update weights
# _ws0 = vmot.update_theta(ws0, theta_star0)
# _ws1 = vmot.update_theta(ws1, theta_star1)
# _ws2 = vmot.update_theta(ws2, theta_star2)
# _ws3 = vmot.update_theta(ws3, theta_star3)
# _ws4 = vmot.update_theta(ws4, theta_star4)

# # new round with new weights
# __model0, __D_evo0, __s_evo0, __H_evo0, __P_evo0 = vmot.mtg_train(_ws0, opt_parameters, model = model0, verbose = 100)
# __model1, __D_evo1, __s_evo1, __H_evo1, __P_evo1 = vmot.mtg_train(_ws1, opt_parameters, model = model1, verbose = 100)
# __model2, __D_evo2, __s_evo2, __H_evo2, __P_evo2 = vmot.mtg_train(_ws2, opt_parameters, model = model2, verbose = 100)
# __model3, __D_evo3, __s_evo3, __H_evo3, __P_evo3 = vmot.mtg_train(_ws3, opt_parameters, model = model3, verbose = 100)
# __model4, __D_evo4, __s_evo4, __H_evo4, __P_evo4 = vmot.mtg_train(_ws4, opt_parameters, model = model4, verbose = 100)

# _model0 = model0 + _model0
# _D_evo0 = D_evo0 + _D_evo0
# _s_evo0 = s_evo0 + _s_evo0
# _H_evo0 = H_evo0 + _H_evo0
# _P_evo0 = P_evo0 + _P_evo0
# _model1 = model1 + _model1
# _D_evo1 = D_evo1 + _D_evo1
# _s_evo1 = s_evo1 + _s_evo1
# _H_evo1 = H_evo1 + _H_evo1
# _P_evo1 = P_evo1 + _P_evo1
# _model2 = model2 + _model2
# _D_evo2 = D_evo2 + _D_evo2
# _s_evo2 = s_evo2 + _s_evo2
# _H_evo2 = H_evo2 + _H_evo2
# _P_evo2 = P_evo2 + _P_evo2
# _model3 = model3 + _model3
# _D_evo3 = D_evo3 + _D_evo3
# _s_evo3 = s_evo3 + _s_evo3
# _H_evo3 = H_evo3 + _H_evo3
# _P_evo3 = P_evo3 + _P_evo3
# _model4 = model4 + _model4
# _D_evo4 = D_evo4 + _D_evo4
# _s_evo4 = s_evo4 + _s_evo4
# _H_evo4 = H_evo4 + _H_evo4
# _P_evo4 = P_evo4 + _P_evo4

# __model0 = model0 + __model0
# __D_evo0 = D_evo0 + __D_evo0
# __s_evo0 = s_evo0 + __s_evo0
# __H_evo0 = H_evo0 + __H_evo0
# __P_evo0 = P_evo0 + __P_evo0
# __model1 = model1 + __model1
# __D_evo1 = D_evo1 + __D_evo1
# __s_evo1 = s_evo1 + __s_evo1
# __H_evo1 = H_evo1 + __H_evo1
# __P_evo1 = P_evo1 + __P_evo1
# __model2 = model2 + __model2
# __D_evo2 = D_evo2 + __D_evo2
# __s_evo2 = s_evo2 + __s_evo2
# __H_evo2 = H_evo2 + __H_evo2
# __P_evo2 = P_evo2 + __P_evo2
# __model3 = model3 + __model3
# __D_evo3 = D_evo3 + __D_evo3
# __s_evo3 = s_evo3 + __s_evo3
# __H_evo3 = H_evo3 + __H_evo3
# __P_evo3 = P_evo3 + __P_evo3
# __model4 = model4 + __model4
# __D_evo4 = D_evo4 + __D_evo4
# __s_evo4 = s_evo4 + __s_evo4
# __H_evo4 = H_evo4 + __H_evo4
# __P_evo4 = P_evo4 + __P_evo4

# results = [_model0, _D_evo0, _s_evo0, _H_evo0, _P_evo0, _model1, _D_evo1, _s_evo1, _H_evo1, _P_evo1, _model2, _D_evo2, _s_evo2, _H_evo2, _P_evo2, _model3, _D_evo3, _s_evo3, _H_evo3, _P_evo3, _model4, _D_evo4, _s_evo4, _H_evo4, _P_evo4, __model0, __D_evo0, __s_evo0, __H_evo0, __P_evo0, __model1, __D_evo1, __s_evo1, __H_evo1, __P_evo1, __model2, __D_evo2, __s_evo2, __H_evo2, __P_evo2, __model3, __D_evo3, __s_evo3, __H_evo3, __P_evo3, __model4, __D_evo4, __s_evo4, __H_evo4, __P_evo4]

# # dump
# _dir = './model_dump/'
# _file = 'quantile_results.pickle'
# _path = _dir + _file
# with open(_path, 'wb') as file:
#     pickle.dump(results, file)
# print('model saved to ' + _path)

# load
_dir = './model_dump/'
_file = 'quantile_results.pickle'
_path = _dir + _file
with open(_path, 'rb') as file:
    results = pickle.load(file)
print('model loaded from ' + _path)
_model0, _D_evo0, _s_evo0, _H_evo0, _P_evo0, _model1, _D_evo1, _s_evo1, _H_evo1, _P_evo1, _model2, _D_evo2, _s_evo2, _H_evo2, _P_evo2, _model3, _D_evo3, _s_evo3, _H_evo3, _P_evo3, _model4, _D_evo4, _s_evo4, _H_evo4, _P_evo4, __model0, __D_evo0, __s_evo0, __H_evo0, __P_evo0, __model1, __D_evo1, __s_evo1, __H_evo1, __P_evo1, __model2, __D_evo2, __s_evo2, __H_evo2, __P_evo2, __model3, __D_evo3, __s_evo3, __H_evo3, __P_evo3, __model4, __D_evo4, __s_evo4, __H_evo4, __P_evo4 = results

# convergence plots
prop_cycle = pl.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

band_size = 1
labels = ['marginal sampling (original)', 'full random (original)', 'grid (quantile)', 'full random (quantile)']
_D_evo_list = [_D_evo0, _D_evo1, _D_evo2, _D_evo3, _D_evo4]
_H_evo_list = [_H_evo0, _H_evo1, _H_evo2, _H_evo3, _H_evo4]
_s_evo_list = [_s_evo0, _s_evo1, _s_evo2, _s_evo3, _s_evo4]
__D_evo_list = [__D_evo0, __D_evo1, __D_evo2, __D_evo3, __D_evo4]
__H_evo_list = [__H_evo0, __H_evo1, __H_evo2, __H_evo3, __H_evo4]
__s_evo_list = [__s_evo0, __s_evo1, __s_evo2, __s_evo3, __s_evo4]

# convergence comparison (D)
# pl.figure(figsize = [12,12])   # plot in two iterations to have a clean legend
# [pl.plot(-np.array(D_evo)) for D_evo in _D_evo_list]
# [pl.plot(-np.array(D_evo), color=colors[i], linestyle=':') for i, D_evo in enumerate(__D_evo_list)]
# pl.legend(labels)
# for D_evo, s_evo in zip(_D_evo_list, _s_evo_list):
#     pl.fill_between(range(len(D_evo)),
#                     -np.array(D_evo) + np.array(band_size * s_evo),
#                     -np.array(D_evo) - np.array(band_size * s_evo), alpha = .3, facecolor = 'grey')
# pl.axhline(ref_value, linestyle=':', color='black')

# convergence comparison (D) (discard 0. marginal sampling)
pl.figure(figsize = [12,12])   # plot in two iterations to have a clean legend
[pl.plot(-np.array(D_evo)) for D_evo in _D_evo_list[1:]]
[pl.plot(-np.array(D_evo), color=colors[i], linestyle=':') for i, D_evo in enumerate(__D_evo_list[1:])]
pl.legend(labels[1:])
for D_evo, s_evo in zip(_D_evo_list[1:], _s_evo_list[1:]):
    pl.fill_between(range(len(D_evo)),
                    -np.array(D_evo) + np.array(band_size * s_evo),
                    -np.array(D_evo) - np.array(band_size * s_evo), alpha = .3, facecolor = 'grey')
pl.axhline(ref_value, linestyle=':', color='black')
pl.axvline(len(_D_evo_list[0])-1, color='grey')










# tests


# convergence comparison (D + H)
# pl.figure(figsize = [12,12])   # plot in two iterations to have a clean legend
# [pl.plot(np.array(D_evo) + np.array(H_evo)) for D_evo, H_evo in zip(D_evo_list, H_evo_list)]
# pl.legend(labels)
# for D_evo, H_evo, s_evo in zip(D_evo_list, H_evo_list, s_evo_list):
#     pl.fill_between(range(len(D_evo)),
#                     np.array(D_evo) + np.array(H_evo) + band_size * np.array(s_evo),
#                     np.array(D_evo) + np.array(H_evo) - band_size * np.array(s_evo), alpha = .3, facecolor = 'grey')
# pl.axhline(ref_value, linestyle=':', color='black')
# model0, _D_evo0, _s_evo0, _H_evo0, _P_evo0 = vmot.mtg_train(ws0, opt_parameters, model = model0, verbose = 100)
# model1, _D_evo1, _s_evo1, _H_evo1, _P_evo1 = vmot.mtg_train(ws1, opt_parameters, model = model1, verbose = 100)
# model2, _D_evo2, _s_evo2, _H_evo2, _P_evo2 = vmot.mtg_train(ws2, opt_parameters, model = model2, verbose = 100)
# model3, _D_evo3, _s_evo3, _H_evo3, _P_evo3 = vmot.mtg_train(ws3, opt_parameters, model = model3, verbose = 100)

# D_evo0 = D_evo0 + _D_evo0
# s_evo0 = s_evo0 + _s_evo0
# H_evo0 = H_evo0 + _H_evo0
# P_evo0 = P_evo0 + _P_evo0

# D_evo1 = D_evo1 + _D_evo1
# s_evo1 = s_evo1 + _s_evo1
# H_evo1 = H_evo1 + _H_evo1
# P_evo1 = P_evo1 + _P_evo1

# D_evo2 = D_evo2 + _D_evo2
# s_evo2 = s_evo2 + _s_evo2
# H_evo2 = H_evo2 + _H_evo2
# P_evo2 = P_evo2 + _P_evo2

# D_evo3 = D_evo3 + _D_evo3
# s_evo3 = s_evo3 + _s_evo3
# H_evo3 = H_evo3 + _H_evo3
# P_evo3 = P_evo3 + _P_evo3

# opt_parameters['macro_epochs'] = 1



opt_parameters['beta_multiplier'] = 10
opt_parameters['beta_multiplier'] = 100
opt_parameters['beta_multiplier'] = 1/10
opt_parameters['beta_multiplier'] = 1/100

# beta_multiplier = 1
D_evo_list_a = D_evo_list.copy()
H_evo_list_a = H_evo_list.copy()

# beta_multiplier = .1
D_evo_list_b = D_evo_list.copy()
H_evo_list_b = H_evo_list.copy()


D_evo_list_1 = D_evo_list_a.copy()
H_evo_list_1 = H_evo_list_a.copy()

D_evo_list_01 = D_evo_list_b.copy()
H_evo_list_01 = H_evo_list_b.copy()

D_evo_list_10 = D_evo_list.copy()
H_evo_list_10 = H_evo_list.copy()

D_evo_list_100 = D_evo_list.copy()
H_evo_list_100 = H_evo_list.copy()

D_evo_list_001 = D_evo_list.copy()
H_evo_list_001 = H_evo_list.copy()



D_evo_list_10 = D_evo_list.copy()



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


