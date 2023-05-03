# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 00:32:56 2023

@author: souzam
"""


    
    # check potential functions individually
    npoints = 201
    
    t = np.linspace(-5, 5, npoints)
    pl.figure(figsize=figsize)
    pl.xlabel('x1, x2')
    pl.ylabel('phi')
    pl.plot(t, h_x1(torch.tensor(t).float().view(len(t), 1)).detach().numpy())
    pl.plot(t, h_x2(torch.tensor(t).float().view(len(t), 1)).detach().numpy())
    pl.legend(['phi_1(x1)', 'phi_2(x2)'])
    
    pl.figure(figsize=figsize)
    pl.xlabel('y1, y2')
    pl.ylabel('psi')
    pl.plot(t, h_y1(torch.tensor(t).float().view(len(t), 1)).detach().numpy())
    pl.plot(t, h_y2(torch.tensor(t).float().view(len(t), 1)).detach().numpy())
    pl.legend(['psi_1(y1)', 'psi_2(y2)'])
    
    x1x2 = torch.tensor(np.vstack([t,t]).T).float().view(len(t), 2)
    pl.figure(figsize=figsize)
    pl.xlabel('x1 (= x2)')
    pl.ylabel('h')
    pl.title('h')
    pl.plot(t, g1(x1x2).detach().numpy())
    pl.plot(t, g2(x1x2).detach().numpy())
    pl.legend(['h1(x1, x2)', 'h2(x1, x2)'])