# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:35:25 2021
@author: souzam
PyTorch implementation of Eckstein and Kupper 2019 - Computation of Optimal Transport...
"""


import numpy as np
from scipy.stats import norm

import vmot_core




# penalty function 
def beta_Lp(x, p, gamma):
    return (1 / gamma) * (1 / p) * torch.pow(torch.relu(gamma * x), p)
 
def beta_L2(x, gamma):
    return beta_Lp(x, 2, gamma)
 
def beta_L2_prime(x, gamma):
    return gamma * torch.relu(x)
    
# class for model for each hj (or gj) to be minimized (rhs of eq 2.8)
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

# train loop
def q_train_loop(sample_loader, phi_list, psi_list, h_list,
                 beta, gamma, optimizer = None, verbose = 0):
    full_size = len(sample_loader.dataset)
    if verbose > 0:
        print('   batch              D              H      deviation              P' + (not optimizer is None) * '                 loss')
        print('--------------------------------------------------------------------' + (not optimizer is None) * '---------------------')
    
    _D = np.array([])   # dual value
    _H = np.array([])   # should converge to zero if distributions are in convex order (for report purposes only)
    _P = np.array([])   # penalty
    # for batch, subsample in enumerate(sample_loader): break
    for batch, subsample in enumerate(sample_loader):
        # subsample expected format:
        # -- X -- | -- Y -- | -- L -- | c | th |
        size, num_cols = subsample.shape
        d = int((num_cols - 2) / 3)
        
        # nn time series with batch_size rows and d columns
        phi = torch.hstack([phi(subsample[:,i].view(size, 1)) for i, phi in enumerate(phi_list)])
        psi = torch.hstack([psi(subsample[:,i+d].view(size, 1)) for i, psi in enumerate(psi_list)])
        h   = torch.hstack([h(subsample[:,:d]) for i, h in enumerate(h_list)])
        
        # cost
        L     = subsample[:,2*d:3*d]
        c     = subsample[:,3*d]
        theta = subsample[:,3*d+1]
        
        # dual value and penalization
        D = (phi + psi).sum(axis=1)   # sum over dimensions
        H = (h * L).sum(axis=1)       # sum over dimensions
        deviation = D + H - c
        P = beta(deviation, gamma)
        
        _D = np.append(_D, D.detach().numpy())
        _H = np.append(_H, H.detach().numpy())
        _P = np.append(_P, P.detach().numpy())
        
        # loss and backpropagation
        p_multiplier = 1.0
        loss = (-D + p_multiplier * theta * P / theta.sum()).mean()
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


# domain-specific program: cost function, Lk functions (k = 1, ..., d) and theta probability
# arguments are quantiles, must be converted to the interval [0,1] and then to the problem
#    domain the using inverse cummulative function

# cross product example

x_normal_scale = [1.0, 1.0]
y_normal_scale = [1.5, 2.0]

def inv_cum_x(xhat, i):
    return norm.ppf(xhat) * x_normal_scale[i]

def inv_cum_y(xhat, i):
    return norm.ppf(xhat) * y_normal_scale[i]

def L_f(q_set):
    # returns an nxd matrix whose i-th column contains L(xhat_i, yhat_i) = y_i - x_i
    qx = q_set[:, :d]
    qy = q_set[:, d:]
    xhat = (2 * qx + 1) / (2 * n)   # a point in the 1-hypercube
    yhat = (2 * qy + 1) / (2 * n)   # a point in the 1-hypercube
    x = np.array([inv_cum_x(xhat[:,i], i) for i in range(d)]).T
    y = np.array([inv_cum_y(yhat[:,i], i) for i in range(d)]).T
    return y - x

def cost_f(q_set):
    # cost = A.x1.x2 + B.y1.y2
    A = 0
    B = 1
    qx = q_set[:, :d]
    qy = q_set[:, d:]
    xhat = (2 * qx + 1) / (2 * n)   # a point in the 1-hypercube
    yhat = (2 * qy + 1) / (2 * n)   # a point in the 1-hypercube
    x = np.array([inv_cum_x(xhat[:,i], i) for i in range(d)]).T
    y = np.array([inv_cum_y(yhat[:,i], i) for i in range(d)]).T
    c = A * x[:,0] * x[:,1] + B * y[:,0] * y[:,1]
    return c

def theta_prob(q_set, coupling = 'independent'):
    # some joint discrete probability on the set of quantiles
    # marginals must be the discrete uniform distribution
    if coupling == 'independent':
        # simplest case, th = 1 / n^2d for all quantiles
        return np.ones(len(q_set)) / len(q_set)
    


# basic script
n = 40
d = 2
print(f'grid size = {n:d}   d = {d:d}   working sample size = {n**(2*d):d}')

# quantile grids for x and y
q_set = np.array(list(itertools.product(*[list(range(n)) for i in range(2 * d)])))

# cross product example
l_set = L_f(q_set)
c_set = cost_f(q_set)
theta = theta_prob(q_set, coupling = 'independent')

# check and build working sample
# working sample format:
# -- X -- | -- Y -- | -- L -- | c | th |
assert q_set.shape == (n ** (2 * d), 2 * d)
assert l_set.shape == (n ** (2 * d), d)
assert c_set.ndim == 1 and len(c_set) == n ** (2 * d)
assert theta.ndim == 1 and len(theta) == n ** (2 * d)
working_sample = np.hstack([q_set, l_set, c_set.reshape(len(theta), 1), theta.reshape(len(theta), 1)])
assert working_sample.shape == (n ** (2 * d), 3 * d + 2)

# loader
batch_size = n ** d   # sqrt of total size; no special formula for this...
shuffle    = True     # must be True to avoid some bias towards the last section of the quantile grid
sample_loader = DataLoader(torch.tensor(working_sample).float(), batch_size = batch_size, shuffle = shuffle)

# modules and optimizers
lr =1e-4
hidden_size = 32
n_hidden_layers = 2
phi_list = nn.ModuleList([PotentialF(1, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d)])
psi_list = nn.ModuleList([PotentialF(1, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d)])
h_list   = nn.ModuleList([PotentialF(d, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size) for i in range(d)])
optimizer = torch.optim.Adam(list(phi_list.parameters()) + list(psi_list.parameters()) + list(h_list.parameters()), lr=lr)



# tests

beta = beta_L2
gamma = 100
verbose = 100
macro_epochs = 10
micro_epochs = 10

D_series = []
s_series = []
H_series = []
P_series = []
t0 = time.time() # timer
for i in range(macro_epochs):
    for j in range(micro_epochs):
        verb = (j + 1 == micro_epochs) * verbose
        print(f'{i+1:4d}, {j+1:3d}')
        if verb:
            print()
        D, s, H, P = q_train_loop(sample_loader, phi_list, psi_list, h_list, beta, gamma, optimizer, verb)
        D_series.append(D)
        s_series.append(D)
        H_series.append(H)
        P_series.append(P)
        if verb:
            print()
            print(f'D   = {D:12.4f}')
            print(f'std = {D:12.4f}')
            print(f'H   = {H:12.4f}')
            print(f'P   = {P:12.4f}')
            print()
t1 = time.time() # timer
print('duration = ' + str(dt.timedelta(seconds=round(t1 - t0))))

pl.plot(D_series)







psi
h
L
c
phi_list[0](subsample[:,0].view(size, 1))
phi_list[1](subsample[:,1].view(size, 1))
psi_list[0](subsample[:,2].view(size, 1))
psi_list[1](subsample[:,3].view(size, 1))
h_list[0](subsample[:,0:2])
h_list[1](subsample[:,0:2])


q_xy = q_set[15:20]







     

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
def train_loop(cost, mu_loader, th_loader,
               phi_x_list, phi_y_list, h_list, beta, gamma,
               optimizer = None, verbose = False):
    
    # f:        cost function to be maximized (primal)
    # phi_list: list of potential functions as neural networks (dual)
    # beta:     penalization function
    
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
        deviation = (cost_th - dual_th)  # array of values
        penalty = torch.mean(beta(deviation, gamma))   # "integral"
        
        _value.append(value.item())
        # _value.append(value.item() + value_h.item())   # if intended to include h_mu
        _penalty.append(penalty.item())
        
        # backpropagation
        if not optimizer is None:
            loss = value + penalty
            # loss = (value + value_h) + penalty   # if intended to include h_mu
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
