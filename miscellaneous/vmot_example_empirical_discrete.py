# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:35:25 2021
@author: souzam
PyTorch implementation of Eckstein and Kupper 2019 - Computation of Optimal Transport...
"""


import numpy as np
import matplotlib.pyplot as pl
import pickle

import vmot_core as vmot
import option_implied_inverse_cdf as empirical


# choose d and marginal distributions
d = 2
# cost function
A = 0
B = 1
def cost_f(x, y):
    # cost = A.x1.x2 + B.y1.y2
    return A * x[:,0] * x[:,1] + B * y[:,0] * y[:,1]

def minus_cost_f(x, y):
    return -cost_f(x, y)

# reference value: unknown
ref_value = None

# common parameters
n = 40   # marginal sample/grid size
full_size = n ** (2 * d)
print(f'full size: {full_size}')
opt_parameters = { 'penalization'    : 'L2',
                   'beta_multiplier' : 10,
                   'gamma'           : 1000,
                   'batch_size'      : n ** d,   # no special formula for this, using sqrt of working sample size
                   'macro_epochs'    : 30,
                   'micro_epochs'    : 20      }

# define marginal strike samples from empirical implied pdf
x1 = empirical.AMZNcombineStrike[:,0]
y1 = empirical.AMZNcombineStrike[:,1]
x2 = empirical.AAPLcombineStrike[:,0]
y2 = empirical.AAPLcombineStrike[:,1]
x1_pdf = empirical.AMZNcombineDensity[:,0]
y1_pdf = empirical.AMZNcombineDensity[:,1]
x2_pdf = empirical.AAPLcombineDensity[:,0]
y2_pdf = empirical.AAPLcombineDensity[:,1]
# clip zero/null strikes
x2_pdf = x2_pdf[x2>0]
y2_pdf = y2_pdf[y2>0]
x2 = x2[x2>0]
y2 = y2[y2>0]
# sizes:
#   |X1| = 12, |Y1| = 12
#   |X2| = 20, |Y2| = 37

# normalize
# note: could simply multiply by the size of the interval between strikes
#       but the sum in the original script wouldn't be exactly one
x1_pdf = x1_pdf / x1_pdf.sum()
x2_pdf = x2_pdf / x2_pdf.sum()
y1_pdf = y1_pdf / y1_pdf.sum()
y2_pdf = y2_pdf / y2_pdf.sum()

# xy_set0 = vmot.xi_yi_to_xy_set([x1, x2], [y1, y2], monotone_x = False)
# theta = vmot.marginal_w_to_w([x1_pdf, x2_pdf], [y1_pdf, y2_pdf], monotone_x = False)
# ws0 = vmot.generate_working_sample(xy_set0, minus_cost_f, theta = theta)
# ws0.shape


if __name__ == '__main__':
    
    # tests - monotone coupling and discrete prob plot
    # xi=[x1[:4],x2[:4]]
    # yi=[y1[:4],y2[4:8]]
    # np.array(xi)
    # np.array(yi)
    # wxi=[np.array([0.2,  0.3,  0.4,  0.1]),
    #      np.array([0.15, 0.3,  0.35, 0.2])]
    # wyi=[np.array([0.3,  0.2,  0.3,  0.2]),
    #      np.array([0.05, 0.35, 0.3,  0.3])]
    
    # ?
    # theta = vmot.marginal_w_to_w([x1_pdf, x2_pdf], [y1_pdf, y2_pdf], monotone_x = False)
    # ws0 = vmot.generate_working_sample(xy_set_m0, minus_cost_f, theta = theta)
    
    # xy_set = vmot.combine_marginals(xi, yi)
    # xy_set_mon = vmot.combine_marginals_monotone(xi, yi)
    # xy_set_m, wm = vmot.combine_marginals_monotone_weighted(xi, yi, wxi, wyi)
    # ws = vmot.generate_working_sample(xy_set_m, minus_cost_f,theta = wm)
    
    # vmot.plot_discrete_prob_2d(ws)
    
    
    # test monotone with full set
    # xy_set_m0, wm0 = vmot.combine_marginals_monotone_weighted([x1, x2], [y1, y2], [x1_pdf, x2_pdf], [y1_pdf, y2_pdf])
    # ws0 = vmot.generate_working_sample(xy_set_m0, minus_cost_f,theta = wm0)
    # vmot.plot_discrete_prob_2d(xy_set_m0[:, 0], xy_set_m0[:, 1], wm0)
    
    # a = vmot.combine_marginals([x1, x2], [y1, y2])
    # b = vmot.combine_marginals_monotone([x1[:12], x2[:12]], [y1[:12], y2[:12]])
    
    # weighted combinations, independent vs monotone
    xy_set_ind, w_ind = vmot.combine_marginals_weighted([x1, x2], [y1, y2], [x1_pdf, x2_pdf], [y1_pdf, y2_pdf])
    ws_ind = vmot.generate_working_sample(xy_set_ind, minus_cost_f,theta = w_ind)
    vmot.plot_discrete_prob_2d(xy_set_ind[:, 0], xy_set_ind[:, 1], w_ind, label = 'Jan 20th', x1label = 'AMZN', x2label = 'AAPL')
    vmot.plot_discrete_prob_2d(xy_set_ind[:, 2], xy_set_ind[:, 3], w_ind, label = 'Feb 17th', x1label = 'AMZN', x2label = 'AAPL')
    
    xy_set_mon, w_mon = vmot.combine_marginals_monotone_weighted([x1, x2], [y1, y2], [x1_pdf, x2_pdf], [y1_pdf, y2_pdf])
    ws_mon = vmot.generate_working_sample(xy_set_mon, minus_cost_f,theta = w_mon)
    vmot.plot_discrete_prob_2d(xy_set_mon[:, 0], xy_set_mon[:, 1], w_mon, label = 'Jan 20th', x1label = 'AMZN', x2label = 'AAPL')
    vmot.plot_discrete_prob_2d(xy_set_mon[:, 2], xy_set_mon[:, 3], w_mon, label = 'Feb 17th', x1label = 'AMZN', x2label = 'AAPL')
    
    # fair convergence comparison - monotone has more epochs to compensate for smaller size
    opt_parameters_mon = opt_parameters.copy()
    opt_parameters_mon['micro_epochs'] = int(opt_parameters['micro_epochs'] * len(ws_ind) / len(ws_mon))
    
    # train and approximate dual
    _model0, _D_evo0, _s_evo0, _H_evo0, _P_evo0 = vmot.mtg_train(ws_ind, opt_parameters, verbose = 100)
    _model1, _D_evo1, _s_evo1, _H_evo1, _P_evo1 = vmot.mtg_train(ws_mon, opt_parameters_mon, verbose = 100)

    
    
w_ind.sum()
w_mon.sum()





# vmot.plot_sample_2d(xy_set0, 'strike sampling')

# run
model0, D_evo0, s_evo0, H_evo0, P_evo0 = vmot.mtg_train(ws0, opt_parameters, verbose = 100)
D0, s0, H0, theta_star0 = vmot.mtg_dual_value(model0, ws0, opt_parameters)

# outputs
print(f'D0 {-D0:8.4f}')

theta_star0 = theta_star0 / theta_star0.sum()
theta_star0 = 0.5 * theta_star0 + 0.5 * np.ones(len(theta_star0)) / len(theta_star0)

# new round with same old weights for comparison
_model0, _D_evo0, _s_evo0, _H_evo0, _P_evo0 = vmot.mtg_train(ws0, opt_parameters, model = model0, verbose = 100)

# update weights
_ws0 = vmot.update_theta(ws0, theta_star0)

# new round with new weights
__model0, __D_evo0, __s_evo0, __H_evo0, __P_evo0 = vmot.mtg_train(_ws0, opt_parameters, model = model0, verbose = 100)

# dump
results = [model0, D_evo0, s_evo0, H_evo0, P_evo0, _model0, _D_evo0, _s_evo0, _H_evo0, _P_evo0, __model0, __D_evo0, __s_evo0, __H_evo0, __P_evo0]
_dir = './model_dump/'
_file = 'empirical_results.pickle'
_path = _dir + _file
with open(_path, 'wb') as file:
    pickle.dump(results, file)
print('model saved to ' + _path)
'''
# load
_dir = './model_dump/'
_file = 'quantile_results.pickle'
_path = _dir + _file
with open(_path, 'rb') as file:
    results = pickle.load(file)
print('model loaded from ' + _path)
model0, D_evo0, s_evo0, H_evo0, P_evo0, _model0, _D_evo0, _s_evo0, _H_evo0, _P_evo0, __model0, __D_evo0, __s_evo0, __H_evo0, __P_evo0 = results
'''

# convergence plots
prop_cycle = pl.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

band_size = 1
labels = ['independent', 'monotone']
_D_evo_list = [D_evo0 + _D_evo0]
_H_evo_list = [H_evo0 + _H_evo0]
_s_evo_list = [s_evo0 + _s_evo0]
__D_evo_list = [D_evo0 + __D_evo0]
__H_evo_list = [H_evo0 + __H_evo0]
__s_evo_list = [s_evo0 + __s_evo0]

_D_evo_list = [_D_evo0, _D_evo1]
_H_evo_list = [_H_evo0, _H_evo1]
_s_evo_list = [_s_evo0, _s_evo1]


# convergence comparison (D)
pl.figure(figsize = [12,12])   # plot in two iterations to have a clean legend
# [pl.plot(-np.array(D_evo)) for D_evo in _D_evo_list]
# [pl.plot(-np.array(D_evo), color=colors[i], linestyle=':') for i, D_evo in enumerate(__D_evo_list)]

pl.legend(labels)
for D_evo, s_evo in zip(_D_evo_list, _s_evo_list):
    pl.fill_between(range(len(D_evo)),
                    -np.array(D_evo) + np.array(band_size * s_evo),
                    -np.array(D_evo) - np.array(band_size * s_evo), alpha = .3, facecolor = 'grey')
# pl.axhline(ref_value, linestyle=':', color='black')
# pl.axvline(len(D_evo0)-1, color='grey')


pl.figure(figsize = [12,12])
pl.plot(-np.array(_D_evo0))
# pl.plot(-np.array(_D_evo1))
pl.plot(-np.array(_D_evo0).repeat(int(len(_D_evo1)/len(_D_evo0))))







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

