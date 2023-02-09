# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:35:25 2021
@author: souzam
PyTorch implementation of Eckstein and Kupper 2019 - Computation of Optimal Transport...
"""


# References:
#    Eckstein and Kupper (2019) "Computation of Optimal Transport and Related Hedging Problems via Penalization and Neural Networks" (EK19)
#    https://github.com/stephaneckstein/transport-and-related
#    https://github.com/stephaneckstein/OT_Comparison

# Common structure
#
#    - SampleDataset is the class for finite samples mu and th respectively from mu0 and theta
#
#    - Phi is the class for potential functions, and implements the neural network
#         . typically there will be one object of this class for each marginal (called phi in general)
#         . in the case of martingale OT there are the martingale potential functions used for penalization (called h in general)
#
#    - Penalization function (beta) and corresponding dual and derivative
#
#    - The train loop has a common structure but the inclusion or not of a martingale penalization are specific to the problem

# Specific structure
#
#    - The sampling of mu reflects the marginals specific to the problem
#    - The sampling of th may simply reflect the marginals specific to the problem or be more sophisticated (like in deterministic coupling)
#    - The cost function defines the objective

# Suggested usage
#
#    1. Dual value
#    - sample mu from mu0 where mu0 satisfies the marginals (used to calculate sum { integral phi_i dmu_i })
#    - sample th from theta satisfying that the solution is absolutely continuous wrt theta (used for penalization)
#    - convert mu and th to torch.tensor's
#    - wrap them in a SampleDataset
#    - create a loader for the dataset, choosing batch_size and shuffle
#    - define the cost function
#    - create one Phi object for each potential function (typical names phi_i for conventional, h_i for martingale)
#    - aggregate the Phi functions in a ModuleList (call it phi_list) and create an optimzer, choosing a solver (eg Adam) and learning rate
#    - define the direction of the optimization and the penalization functions (currently available: betaL2), NN hidden size etc
#    - run iterate calls to the train_loop to optimize the weights of the Phi's
#    - dual value = integral_phi_dmu
#
#    2. pi_hat
#    - sample_th from theta
#    - calculate f and h for sample_th
#    - define beta_prime from the penalization function beta (currently: beta_L2_prime)
#    - draw points from sample_th with probability proportional to beta_prime
#    - plot points

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'   # pytorch version issues

# class for auxiliary dataset
class SampleDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
       
    def __len__(self):
        return len(self.X)
       
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
 
def generate_tensors(sample_mu_X, sample_mu_Y, sample_th_X, sample_th_Y):
    t_mu_X = torch.tensor(sample_mu_X).float()
    t_mu_Y = torch.tensor(sample_mu_Y).float()
    t_th_X = torch.tensor(sample_th_X).float()
    t_th_Y = torch.tensor(sample_th_Y).float()
    return t_mu_X, t_mu_Y, t_th_X, t_th_Y

def generate_loaders(sample_mu_X, sample_mu_Y, sample_th_X, sample_th_Y, batch_size, shuffle = True):
    t_mu_X, t_mu_Y, t_th_X, t_th_Y = generate_tensors(sample_mu_X, sample_mu_Y, sample_th_X, sample_th_Y)
    mu_dataset = SampleDataset(t_mu_X, t_mu_Y)
    th_dataset = SampleDataset(t_th_X, t_th_Y)
    mu_loader = DataLoader(mu_dataset, batch_size = batch_size, shuffle = shuffle)
    th_loader = DataLoader(th_dataset, batch_size = batch_size, shuffle = shuffle)
    return mu_loader, th_loader
    
# class for model for each hj (or gj) to be minimized (rhs of eq 2.8)
class Phi(nn.Module):
 
    def __init__(self, input_dimension, n_hidden_layers = 2, hidden_size = 32):
        super(Phi, self).__init__()
        layers = [nn.Linear(input_dimension, hidden_size), nn.ReLU()]
        for i in range(n_hidden_layers):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers += [nn.Linear(hidden_size, 1)]
        self.hi = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.hi(x)
 
 
# penalty function 
def beta_Lp(x, p, gamma):
    return (1 / gamma) * (1 / p) * torch.pow(torch.relu(gamma * x), p)
 
def beta_L2(x, gamma):
    return beta_Lp(x, 2, gamma)
    # beta(x) = gamma * 1/2 * x^2
 
def beta_L2_prime(x, gamma):
    return gamma * torch.relu(x)
    # beta'(x) = gamma * (x)+
    

# main function
def train_loop(cost, primal_obj, mu_loader, th_loader,
               phi_x_list, phi_y_list, h_list, beta, gamma,
               optimizer = None, verbose = False):
    
    # NOTE: primal_obj=min is unstable; use primal_obj=max with a negative of the cost function instead
    
    # f:      cost function (primal)
    # h_list: list of potential functions (dual)
    # beta:   penalization function
    
    # primal maximization => dual minimization => sign is positive
    # primal minimization => dual maximization => sign is negative
    sign = 1 if primal_obj == 'max' else -1
    
    full_size = len(mu_loader.dataset)
    if verbose:
        if optimizer is None:
            print('       sum_mu[phi]           sum_mu[h]     sum_th[penalty]           deviation')
        else:
            print('       sum_mu[phi]           sum_mu[h]     sum_th[penalty]                loss')
    _value     = []
    _penalty   = []
    # for batch, ((_mu_X, _mu_Y), (_th_X, _th_Y)) in enumerate(zip(mu_loader, th_loader)): break   # test mode only
    for batch, ((_mu_X, _mu_Y), (_th_X, _th_Y)) in enumerate(zip(mu_loader, th_loader)):
        size, d = _mu_X.shape
        
        # value integral, on mu
        phi_x_values_mu = [phi(_mu_X[:,i].view(size, 1))[:,0] for i, phi in enumerate(phi_x_list)]
        phi_y_values_mu = [phi(_mu_Y[:,i].view(size, 1))[:,0] for i, phi in enumerate(phi_y_list)]
        phi_x_mu  = sum(phi_x_values_mu)         # array of values
        phi_y_mu  = sum(phi_y_values_mu)         # array of values
        value = torch.mean(phi_x_mu) + torch.mean(phi_y_mu)  # "integral"
        
        # for show only (we will only look at h in the penalization integral below)
        h_values_mu = [phi(_mu_X.view(size, d))[:,0] * (_mu_Y[:,i] - _mu_X[:,i]) for i, phi in enumerate(h_list)]
        h_mu    = sum(h_values_mu)   # array of values
        value_h = torch.mean(h_mu)   # "integral"
        
        # penalization integral, on th
        cost_th = cost(_th_X, _th_Y)
        phi_x_values_th = [phi(_th_X[:,i].view(size, 1))[:,0] for i, phi in enumerate(phi_x_list)]
        phi_y_values_th = [phi(_th_Y[:,i].view(size, 1))[:,0] for i, phi in enumerate(phi_y_list)]
        h_values_th = [phi(_th_X.view(size, d))[:,0] * (_th_Y[:,i] - _th_X[:,i]) for i, phi in enumerate(h_list)]
        phi_x_th = sum(phi_x_values_th)         # array of values
        phi_y_th = sum(phi_y_values_th)         # array of values
        h_th     = sum(h_values_th)             # array of values
        dual_th = phi_x_th + phi_y_th + h_th    # array of values
        deviation = sign * (cost_th - dual_th)  # array of values
        penalty = torch.mean(beta(deviation, gamma))   # "integral"
        
        _value.append(value.item())
        # _value.append(value.item() + value_h.item())   # if intended to include h_mu
        _penalty.append(penalty.item())
        
        # backpropagation
        if not optimizer is None:
            loss = (sign * value) + penalty
            # loss = (sign * (value + value_h)) + penalty   # if intended to include h_mu
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # iteration report
        if verbose:
            c = (batch + 1) * size
            if batch == 0 or c % 100000 == 0:
                if optimizer is None:
                    print(f'{value.item():18.4f}' + \
                        f'  {value_h.item():18.4f}' + \
                        f'  {penalty.item():18.4f}' + \
                        f'  {torch.mean(deviation).item():18.4f}')
                else:
                    print(f'{value.item():18.4f}' + \
                        f'  {value_h.item():18.4f}' + \
                        f'  {penalty.item():18.4f}' + \
                        f'  {loss.item():18.4f}    [{c:>7d}/{full_size:>7d}]')
                        
                    
    return np.mean(_value), np.std(_value), np.mean(_penalty)
