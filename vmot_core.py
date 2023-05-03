# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:35:25 2021
@author: souzam
Dual approximation of VMOT using neural networks
References
- Hiew, Lim Pass, Souza "Modularity, Convex conjugates and VMOT" (in development)
- Eckstein and Kupper "Computation of Optimal Transport" (2021)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import itertools
import torch
from torch import nn
from torch.utils.data import DataLoader

import datetime as dt, time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'   # pytorch version issues


# penalty function and derivative
def beta_Lp(x, p, gamma):
    return (1 / gamma) * (1 / p) * torch.pow(torch.relu(gamma * x), p)
 
def beta_L2(x, gamma):
    return beta_Lp(x, 2, gamma)
 
def beta_L2_prime(x, gamma):
    return gamma * torch.relu(x)
    

# base class for each potential function phi_i, psi_i or h_i to be maximized
class PotentialF(nn.Module):
    def __init__(self, input_dimension, n_hidden_layers = 2, hidden_size = 32):
        super(PotentialF, self).__init__()
        layers = [nn.Linear(input_dimension, hidden_size), nn.ReLU()]
        for i in range(n_hidden_layers):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers += [nn.Linear(hidden_size, 1)]
        self.hi = nn.Sequential(*layers)
    def forward(self, x):
        return self.hi(x)

# extract from working sample
# format:
# -- (Q)X -- | -- (Q)Y -- | -- L -- | c | th |
def mtg_parse(model, sample):
    
    phi_list, psi_list, h_list = model
    size, num_cols = sample.shape
    nx, ny = len(phi_list), len(psi_list)    # input lengths (depend on dimensionality)
    d = num_cols - nx - ny - 2
    
    # extract from the working sample
    X     = sample[:, : nx]
    Y     = sample[:, nx : nx + ny]
    L     = sample[:, nx + ny : nx + ny + d]
    c     = sample[:, nx + ny + d]
    theta = sample[:, nx + ny + d + 1]
    
    # calculate using model
    phi = torch.hstack([phi(X[:, i].view(size, 1)) for i, phi in enumerate(phi_list)])
    psi = torch.hstack([psi(Y[:, j].view(size, 1)) for j, psi in enumerate(psi_list)])
    h   = torch.hstack([h(X) for h in h_list])
    
    return phi, psi, h, L, c, theta
    
# train loop
def mtg_train_loop(model, working_loader, beta, beta_multiplier, gamma, optimizer = None, verbose = 0):
    
    #   Primal:          min C
    #   Dual:            max D + H  st  D + H <= C
    #   Penalized dual:  min -(D + H) + b(D + H - C)
    
    full_size = len(working_loader.dataset)
    if verbose > 0:
        print('   batch              D              H      deviation              P' + (not optimizer is None) * '                 loss')
        print('--------------------------------------------------------------------' + (not optimizer is None) * '---------------------')
    
    _D = np.array([])   # dual value
    _H = np.array([])   # mtg component (for report purposes only) - must converge to zero when (mu) <= (nu)
    _P = np.array([])   # penalty
    
    # for batch, sample in enumerate(sample_loader): break
    for batch, sample in enumerate(working_loader):
        
        # time series of dual value, mtg component and penalization
        phi, psi, h, L, c, theta = mtg_parse(model, sample)
        D = phi.sum(axis=1) + psi.sum(axis=1)   # sum over dimensions
        H = (h * L).sum(axis=1)       # sum over dimensions
        deviation = D + H - c
        P = beta(deviation, gamma)
        
        _D = np.append(_D, D.detach().numpy())
        _H = np.append(_H, H.detach().numpy())
        _P = np.append(_P, P.detach().numpy())
        
        # loss and backpropagation
        # loss = (-D + b_multiplier * P).mean()
        loss = -(1 / full_size) * D.sum() + beta_multiplier * (theta * P).sum()
        if not optimizer is None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # iteration report
        parsed = len(_D)
        if verbose > 0 and (parsed == full_size or (batch+1) % verbose == 0):
                print(f'{batch+1:8d}' + 
                      f'   {D.mean().item():12.4f}' +
                      f'   {H.mean().item():12.4f}' +
                      f'   {deviation.mean().item():12.4f}' +
                      f'   {P.mean().item():12.4f}' +
                      (not optimizer is None) * f'   {loss.item():18.4f}' +
                      f'    [{parsed:>7d}/{full_size:>7d}]')
        
    return _D.mean(), _D.std(), _H.mean(), _P.mean()

def mtg_train(working_sample, opt_parameters, model = None, monotone = False, verbose = False):
    
    # check inputs
    n, num_cols = working_sample.shape
    if monotone:
        d = int((num_cols - 3) / 2)
    else:
        d = int((num_cols - 2) / 3)
    # print('d', d)
    if 'penalization' in opt_parameters.keys() and opt_parameters['penalization'] != 'L2':
        print('penalization not implemented: ' + opt_parameters['penalization'])
        return
    beta            = beta_L2                   # L2 penalization
    beta_multiplier = opt_parameters['beta_multiplier']
    gamma           = opt_parameters['gamma']
    batch_size      = opt_parameters['batch_size']
    macro_epochs    = opt_parameters['macro_epochs']
    micro_epochs    = opt_parameters['micro_epochs']
    
    # loader
    shuffle        = True     # must be True to avoid some bias towards the last section of the quantile grid
    working_sample = torch.tensor(working_sample).float()
    working_loader = DataLoader(working_sample, batch_size = batch_size, shuffle = shuffle)
    
    # modules and optimizers
    lr =1e-4
    hidden_size = 32
    n_hidden_layers = 2
    if model is None:
        if monotone:
            phi_list = nn.ModuleList([PotentialF(1, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size)])
            psi_list = nn.ModuleList([PotentialF(1, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d)])
            h_list   = nn.ModuleList([PotentialF(1, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d)])
        else:
            phi_list = nn.ModuleList([PotentialF(1, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d)])
            psi_list = nn.ModuleList([PotentialF(1, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d)])
            h_list   = nn.ModuleList([PotentialF(d, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d)])
        model = nn.ModuleList([phi_list, psi_list, h_list])
    # else:
    #     phi_list, psi_list, h_list = model
    # optimizer = torch.optim.Adam(list(phi_list.parameters()) + list(psi_list.parameters()) + list(h_list.parameters()), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # iterative calls to train_loop
    D_evolution = []
    s_evolution = []
    H_evolution = []
    P_evolution = []
    if verbose > 0:
        t0 = time.time() # timer
    for i in range(macro_epochs):
        for j in range(micro_epochs):
            verb = (j + 1 == micro_epochs) * verbose
            if verbose > 0 and j+1 % 10 == 0:
                print(f'{i+1:4d}, {j+1:3d}')
            if verb:
                print()
            D, s, H, P = mtg_train_loop(model, working_loader, beta, beta_multiplier, gamma, optimizer, verb)
            D_evolution.append(D)
            s_evolution.append(s)
            H_evolution.append(H)
            P_evolution.append(P)
            if verb:
                print('\nmeans')
                print(f'   D   = {D:12.4f}')
                print(f'   std = {s:12.4f}')
                print(f'   H   = {H:12.4f}')
                print(f'   P   = {P:12.4f}\n')
    if verbose > 0:
        t1 = time.time() # timer
        print('duration = ' + str(dt.timedelta(seconds=round(t1 - t0))))
        
    return model, D_evolution, s_evolution, H_evolution, P_evolution
    
def mtg_dual_value(model, working_sample, opt_parameters):
    if 'penalization' in opt_parameters.keys() and opt_parameters['penalization'] != 'L2':
        print('penalization not implemented: ' + opt_parameters['penalization'])
        return
    beta_prime   = beta_L2_prime   # first derivative of L2 penalization function
    gamma        = opt_parameters['gamma']
    phi_list, psi_list, h_list = model
    
    working_sample = torch.tensor(working_sample).float()
    phi, psi, h, L, c, theta = mtg_parse(model, working_sample)
    D = (phi + psi).sum(axis=1)   # sum over dimensions
    H = (h * L).sum(axis=1)       # sum over dimensions
    deviation = D + H - c
    theta_star = (theta * beta_prime(deviation, gamma))
    
    return D.mean().item(), D.std().item(), H.mean().item(), theta_star.detach().numpy()

# working sample format:
# -- X -- | -- Y -- | -- L -- | c | th |
def generate_working_sample(xy_set, cost_f,
                            theta = None,
                            uniform_theta = False):
    size, num_cols = xy_set.shape
    d = int(num_cols / 2)
    x = xy_set[:, :d]
    y = xy_set[:, d:]
    l = y - x          # each column i has (yi - xi)
    c = cost_f(x, y)   # vector of costs
    if theta is None:
        if uniform_theta:
            theta = np.ones(size) / size
        else:
            print('a theta probability must be specified')
            return None
    working_sample = np.hstack([xy_set, l, c.reshape(size, 1), theta.reshape(size, 1)])
    assert working_sample.shape[1] == 3 * d + 2
    return working_sample

# working sample format:
# -- QX -- | -- QY -- | -- L -- | c | th |
def generate_working_sample_q(q_set, inv_cum_xi, inv_cum_yi, cost_f,
                              theta = None,
                              uniform_theta = False):
    size, num_cols = q_set.shape
    d = int(num_cols / 2)
    qx = q_set[:, :d]
    qy = q_set[:, d:]
    x = np.array([inv_cum_xi(qx[:,i], i) for i in range(d)]).T
    y = np.array([inv_cum_yi(qy[:,i], i) for i in range(d)]).T
    l = y - x          # each column i has (yi - xi)
    c = cost_f(x, y)   # vector of costs
    if theta is None:
        if uniform_theta:
            theta = np.ones(size) / size
        else:
            print('a theta probability must be specified')
            return None
    working_sample = np.hstack([q_set, l, c.reshape(size, 1), theta.reshape(size, 1)])
    return working_sample

def generate_working_sample_q_monotone(q_set, inv_cum_x, inv_cum_yi, cost_f,
                              theta = None,
                              uniform_theta = False):
    size, num_cols = q_set.shape
    d = int(num_cols - 1)
    qx = q_set[:, 0]        # n x 1
    qy = q_set[:, 1:d+1]    # n x d
    x = inv_cum_x(qx)       # n x d
    y = np.array([inv_cum_yi(qy[:,i], i) for i in range(d)]).T   # n x d
    l = y - x          # each column i has (yi - xi)
    c = cost_f(x, y)   # vector of costs
    if theta is None:
        if uniform_theta:
            theta = np.ones(size) / size
        else:
            print('a theta probability must be specified')
            return None
    working_sample = np.hstack([q_set, l, c.reshape(size, 1), theta.reshape(size, 1)])
    return working_sample

# utils - monotonically couple a pair of discrete probabilities
def couple(X1, X2, w1, w2):
    # X1 and X2 should be sorted before calling
    # will return X, w
    #   [ X_ , w_ ] <-- single row calculated in this step
    #   [ _X , _w ] <-- stack calculated recursively
    assert len(X1) == len(w1) and len(X2) == len(w2), 'weight length error'
    assert np.isclose(sum(w1), sum(w2))
    if len(X1) == 1 and len(X2) == 1:
        assert np.isclose(w1[0], w2[0]), 'weight matching error'
        return np.array([X1[0], X2[0]]), w1[0]
    if np.isclose(w1[0], w2[0]):
        X_ = np.array([X1[0], X2[0]]).T
        w_  = w1[0]
        _X, _w = couple(X1[1:], X2[1:], w1[1:], w2[1:])
    elif w1[0] < w2[0]:
        X_ = np.array([X1[0], X2[0]])
        w_  = w1[0]
        _w2 = w2.copy()
        _w2[0] = w2[0] - w1[0]
        _X, _w = couple(X1[1:], X2, w1[1:], _w2)
    elif w1[0] > w2[0]:
        X_ = np.array([X1[0], X2[0]])
        w_  = w2[0]
        _w1 = w1.copy()
        _w1[0] = w1[0] - w2[0]
        _X, _w = couple(X1, X2[1:], _w1, w2[1:])
    X = np.vstack([X_, _X])
    w  = np.vstack([w_, _w])
    return X, w
    
# utils - generate sample set from marginal samples - all possible combinations
def combine_marginals(xi, yi):
    marginals_list = xi + yi
    xy_set = np.array(list(itertools.product(*marginals_list)))
    return xy_set

def combine_marginals_monotone(xi, yi):
    # xi's must be of same size and be ordered before function call
    assert np.min([len(x) for x in xi]) == np.max([len(x) for x in xi]), 'marginals must have same size'
    xi = np.array(xi).T
    marginals_list = [xi] + yi
    inhomogeneous_xy = list(itertools.product(*marginals_list))
    xy_set = np.vstack([np.hstack(inhomogeneous_xy[t]) for t in range(len(inhomogeneous_xy))])
    return xy_set

def combine_marginals_weighted(xi, yi, wxi, wyi):
    marginals_list = xi + yi
    xy_set = np.array(list(itertools.product(*marginals_list)))
    w_list = wxi + wyi
    w = np.array(list(itertools.product(*w_list)))
    w = w.prod(axis=1)
    return xy_set, w

def combine_marginals_monotone_weighted(xi, yi, wxi, wyi):
    # only for d = 2
    assert len(xi) == 2, 'restricted to d=2, higher dimension not implemented'
    X1 = xi[0]
    X2 = xi[1]
    w1 = wxi[0]
    w2 = wxi[1]
    X, wx = couple(X1, X2, w1, w2)
    
    marginals_list = [X] + yi
    inhomogeneous_xy = list(itertools.product(*marginals_list))
    xy_set = np.vstack([np.hstack(inhomogeneous_xy[t]) for t in range(len(inhomogeneous_xy))])
    
    w_list = [wx] + wyi
    inhomogeneous_w = list(itertools.product(*w_list))
    w = np.vstack([np.hstack(inhomogeneous_w[t]) for t in range(len(inhomogeneous_w))])
    w = w.prod(axis=1)
    
    return xy_set, w

def xi_yi_to_xy_set(xi_list, yi_list, monotone_x = False):
    if monotone_x:
        print('not implemented')
        return None
        # order xi's, shuffle together and separate again
        # xi_sample_list = [np.sort(xi) for xi in xi_sample_list]
        # x_set = np.vstack(xi_sample_list).T
        # np.random.shuffle(x_sample)
        # # xi_sample_list = [x_sample[:,i] for i in range(x_sample.shape[1])]
        # # couple each instance of x as a block with all possible combinations of yi's
        # # y_combination = np.array(list(itertools.product(*yi_sample_list)))
        # marginals_list = list(x_sample) + yi_sample_list
        # xy_set = np.array(list(itertools.product(*marginals_list)))
    else:
        # all combinations of x cross y
        marginals_list = xi_list + yi_list
        xy_set = np.array(list(itertools.product(*marginals_list)))
        return xy_set

# utils - generate probability from marginal probabilities
def marginal_w_to_w(xi_w_list, yi_w_list, monotone_x = False):
    if monotone_x:
        print('not implemented')
        return None
        # order xi's, shuffle together and separate again
        # xi_sample_list = [np.sort(xi) for xi in xi_sample_list]
        # x_set = np.vstack(xi_sample_list).T
        # np.random.shuffle(x_sample)
        # # xi_sample_list = [x_sample[:,i] for i in range(x_sample.shape[1])]
        # # couple each instance of x as a block with all possible combinations of yi's
        # # y_combination = np.array(list(itertools.product(*yi_sample_list)))
        # marginals_list = list(x_sample) + yi_sample_list
        # xy_set = np.array(list(itertools.product(*marginals_list)))
    else:
        # all combinations of x cross y
        marginals_list = xi_w_list + yi_w_list
        w = np.array(list(itertools.product(*marginals_list)))
        return w.prod(axis=1)
      
# utils - generate sample set from grid
def grid_to_q_set(n, d, monotone_x = False):
    # all combinations of quantiles on x cross y
    if monotone_x:
        # pending test
        n_grid = np.array(list(itertools.product(*[list(range(n)) for i in range(d+1)])))
        n_grid = np.hstack([np.tile(n_grid[:,0], (d-1, 1)).T, n_grid])   # repeat the xi's
    else:
        n_grid = np.array(list(itertools.product(*[list(range(n)) for i in range(2 * d)])))
    q_set = (2 * n_grid + 1) / (2 * n)   # points in the d-hypercube
    return q_set

def update_theta(working_sample, theta_star):
    working_sample[:, -1] = theta_star
    return working_sample

# utils - 2d plot
def plot_sample_2d(sample, label='sample'):
    figsize = [12,12]
    X1, X2, Y1, Y2 = sample[:,0], sample[:,1], sample[:,2], sample[:,3]
    pl.figure(figsize=figsize)
    pl.title(label)
    pl.axis('equal')
    pl.xlabel('X,Y 1')
    pl.ylabel('X,Y 2')
    pl.scatter(Y1, Y2, alpha=.05)
    pl.scatter(X1, X2, alpha=.05)
    pl.legend(['Y sample', 'X sample'])
    
# utils - plot discrete probability (separated by x and y)
# note: proportions are not working properly, adjust manually before reporting
def plot_discrete_prob_2d(x1, x2, w, label='Probability', x1label = 'x1', x2label = 'X2'):
    # use pandas to reindex and group by
    WS = pd.DataFrame({'x1': x1, 'x2': x2, 'w': w})
    xw = WS[['x1', 'x2', 'w']].groupby(['x1', 'x2']).sum().reset_index()
    piv_x = xw.pivot(index='x1', columns='x2', values='w')
    x1 = piv_x.index
    x2 = piv_x.columns
    xz = piv_x.T.values
    x2_cum = piv_x.sum(axis=0)
    x1_cum = piv_x.sum(axis=1)
    # yw = WS[[2, 3, 7]].groupby([2, 3]).sum().reset_index()
    # piv_y = yw.pivot(index=2, columns=3, values=7)
    # y1 = piv_y.index
    # y2 = piv_y.columns
    # yz = piv_y.T.values
    # y2_cum = piv_y.sum(axis=0)
    # y1_cum = piv_y.sum(axis=1)
    
    min_z = 0.0
    max_z = np.max([np.nanmax(x2_cum), np.nanmax(x1_cum)])
    # max_z = np.max([np.nanmax(x2_cum), np.nanmax(x1_cum), np.nanmax(y2_cum), np.nanmax(y1_cum)])
    x1_step = (x1[len(x1)-1] - x1[0]) / (len(x1) - 1)
    x1_grid = np.append(x1, x1[len(x1)-1] + x1_step) - 0.5 * x1_step
    x2_step = (x2[len(x2)-1] - x2[0]) / (len(x2) - 1)
    x2_grid = np.append(x2, x2[len(x2)-1] + x2_step) - 0.5 * x2_step
    # y1_step = (y1[len(y1)-1] - y1[0]) / (len(y1) - 1)
    # y1_grid = np.append(y1, y1[len(y1)-1] + y1_step) - 0.5 * y1_step
    # y2_step = (y2[len(y2)-1] - y2[0]) / (len(y2) - 1)
    # y2_grid = np.append(y2, y2[len(y2)-1] + y2_step) - 0.5 * y2_step
    
    cmap = pl.colormaps['Reds']
    fig = pl.figure(figsize=[12,12])
    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
    (ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col', sharey='row')
    
    ax2.set_title(label)
    # fig.delaxes(ax3)
    ax3.remove()
    
    # ax1.set_aspect('equal')
    ax1.pcolormesh(x1[:2], x2_grid, x2_cum.values.reshape([len(x2_cum), 1]), vmin=min_z, vmax=max_z, cmap=cmap)
    ax1.get_xaxis().set_visible(False)
    ax1.set_ylabel(x2label)
    ax1.set_aspect('equal', anchor='SE')
    
    ax2.set_aspect('equal', anchor='SW')
    im = ax2.pcolormesh(x1, x2, xz, vmin=min_z, vmax=max_z, shading='auto', cmap=cmap)
    fig.colorbar(im)
    
    # ax4.set_aspect('equal')
    ax4.set_aspect('equal', anchor='NW')
    ax4.pcolormesh(x1_grid, x2[:2], x1_cum.values.reshape([1, len(x1_cum)]), vmin=min_z, vmax=max_z, cmap=cmap)
    ax4.get_yaxis().set_visible(False)
    ax4.set_xlabel(x1label)
    
    for ax in fig.get_axes():
        ax.label_outer()
    
    # x
    '''
    ax_main = pl.subplot(2,2,2)
    ax_main.set_title('x')
    ax_main.set_aspect('equal')
    ax_main.pcolormesh(x1, x2, xz, vmin=min_z, vmax=max_z, shading='auto', cmap=cmap)
    im = ax_main.pcolormesh(x1, x2, xz, vmin=min_z, vmax=max_z, shading='auto', cmap=cmap)
    pl.colorbar(im, ax=ax_main)
    
    ax_side = pl.subplots(2,2,1, sharey=ax_main)
    ax_side.set_aspect('equal')
    ax_side.pcolormesh(x1[:2], x2_grid, x2_cum.values.reshape([len(x2_cum), 1]), vmin=min_z, vmax=max_z, cmap=cmap)
    ax_side.get_xaxis().set_visible(False)
    
    ax_botm = pl.subplots(2,2,1, sharex=ax_main)
    ax_botm.set_aspect('equal')
    ax_botm.pcolormesh(x1_grid, x2[:2], x1_cum.values.reshape([1, len(x1_cum)]), vmin=min_z, vmax=max_z, cmap=cmap)
    ax_botm.get_yaxis().set_visible(False)
    ''' '''
    fig, ax = pl.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=figsize)
    ax[0][0].sharey = ax[0][1]
    ax[1][1].sharex = ax[0][1]
    
    ax[0][0].set_aspect('equal')
    ax[0][0].pcolormesh(x1[:2], x2_grid, x2_cum.values.reshape([len(x2_cum), 1]), vmin=min_z, vmax=max_z, cmap=cmap)
    ax[0][0].get_xaxis().set_visible(False)
    # ax[0][0].spines['top'].set_visible(False)
    # ax[0][0].spines['right'].set_visible(False)
    # ax[0][0].spines['bottom'].set_visible(False)
    
    ax[0][1].set_title('x')
    ax[0][1].set_aspect('equal')
    ax[0][1].pcolormesh(x1, x2, xz, vmin=min_z, vmax=max_z, shading='auto', cmap=cmap)
    im = ax[0][1].pcolormesh(x1, x2, xz, vmin=min_z, vmax=max_z, shading='auto', cmap=cmap)
    fig.colorbar(im, ax=ax[0][1])
    
    ax[1][1].set_aspect('equal')
    ax[1][1].pcolormesh(x1_grid, x2[:2], x1_cum.values.reshape([1, len(x1_cum)]), vmin=min_z, vmax=max_z, cmap=cmap)
    ax[1][1].get_yaxis().set_visible(False)
    # ax[1][1].spines['top'].set_visible(False)
    # ax[1][1].spines['right'].set_visible(False)
    # ax[1][1].spines['left'].set_visible(False)
    
    
    fig.delaxes(ax[1][0])
'''     '''
    figsize = [24,12]
    cmap = pl.colormaps['Reds']
    fig, ax = pl.subplots(2, 4, sharey=True, figsize=figsize)
    
    ax[0][0].set_aspect('equal')
    ax[0][0].set_title('x2')
    ax[0][0].pcolormesh(x1[:2], x2_grid, x2_cum.values.reshape([len(x2_cum), 1]), vmin=min_z, vmax=max_z, cmap=cmap)
    ax[0][0].get_xaxis().set_visible(False)
    ax[0][0].spines['top'].set_visible(False)
    ax[0][0].spines['right'].set_visible(False)
    ax[0][0].spines['bottom'].set_visible(False)
    
    ax[0][1].set_title('x')
    ax[0][1].set_aspect('equal')
    ax[0][1].pcolormesh(x1, x2, xz, vmin=min_z, vmax=max_z, shading='auto', cmap=cmap)
    # im = ax[0][1].pcolormesh(x1, x2, xz, vmin=min_z, vmax=max_z, shading='auto', cmap=cmap)
    # fig.colorbar(im, ax=ax[0])
    
    ax[0][2].set_aspect('equal')
    ax[0][2].set_title('y2')
    ax[0][2].pcolormesh(y1[:2], y2_grid, y2_cum.values.reshape([len(y2_cum), 1]), vmin=min_z, vmax=max_z, cmap=cmap)
    ax[0][2].get_xaxis().set_visible(False)
    ax[0][2].spines['top'].set_visible(False)
    ax[0][2].spines['right'].set_visible(False)
    ax[0][2].spines['bottom'].set_visible(False)
    
    ax[0][3].set_title('y')
    ax[0][3].set_aspect('equal')
    im = ax[0][3].pcolormesh(y1, y2, yz, vmin=min_z, vmax=max_z, shading='auto', cmap=cmap)
    
    fig.colorbar(im, ax=ax[0][3])
    ax[1][1].set_aspect('equal')
    ax[1][1].set_title('x1')
    ax[1][1].pcolormesh(x1_grid, x2[:2], x1_cum.values.reshape([1, len(x1_cum)]), vmin=min_z, vmax=max_z, cmap=cmap)
    ax[1][1].get_yaxis().set_visible(False)
    ax[1][1].spines['top'].set_visible(False)
    ax[1][1].spines['right'].set_visible(False)
    ax[1][1].spines['left'].set_visible(False)
    '''
    
    
    
    # fig, ax = pl.subplots(1, 2, sharey=True, figsize=figsize)
    # ax[0].set_title('x')
    # ax[0].set_aspect('equal')
    # im = ax[0].pcolormesh(x1, x2, xz, vmin=min_z, vmax=max_z, shading='auto', cmap=cmap)
    # # fig.colorbar(im, ax=ax[0])
    # ax[1].set_title('y')
    # ax[1].set_aspect('equal')
    # im = ax[1].pcolormesh(y1, y2, yz, vmin=min_z, vmax=max_z, shading='auto', cmap=cmap)
    # fig.colorbar(im, ax=ax[1])

    # from pylab import *
    # hist2d(x, y, bins=40, norm=LogNorm())
