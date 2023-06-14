# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:12:35 2023

@author: souzam
"""


# def convergence_plot_std(value_series_list, std_series_list, labels, ref_value = None):
#     pl.figure(figsize = [10,10])   # plot in two iterations to have a clean legend
#     for v, std in zip(value_series_list, std_series_list):
#         pl.plot(v)
#     pl.legend(labels)
#     for v, std in zip(value_series_list, std_series_list):
#         pl.fill_between(range(len(v)), v + std, v - std, alpha = .5, facecolor = 'grey')
#     if not ref_value is None:
#         pl.axhline(ref_value, linestyle=':', color='black')
#     pl.show()

# utils - generate sample set from marginal samples - all possible combinations
# def combine_marginals(xi, yi):
#     marginals_list = xi + yi
#     xy_set = np.array(list(itertools.product(*marginals_list)))
#     return xy_set

# def combine_marginals_monotone(xi, yi):
#     # xi's must be of same size and be ordered before function call
#     assert np.min([len(x) for x in xi]) == np.max([len(x) for x in xi]), 'marginals must have same size'
#     xi = np.array(xi).T
#     marginals_list = [xi] + yi
#     inhomogeneous_xy = list(itertools.product(*marginals_list))
#     xy_set = np.vstack([np.hstack(inhomogeneous_xy[t]) for t in range(len(inhomogeneous_xy))])
#     return xy_set

# def combine_marginals_weighted(xi, yi, wxi, wyi):
#     marginals_list = xi + yi
#     xy_set = np.array(list(itertools.product(*marginals_list)))
#     w_list = wxi + wyi
#     w = np.array(list(itertools.product(*w_list)))
#     w = w.prod(axis=1)
#     return xy_set, w

# def combine_marginals_monotone_weighted(xi, yi, wxi, wyi):
#     # only for d = 2
#     assert len(xi) == 2, 'restricted to d=2, higher dimension not implemented'
#     X1 = xi[0]
#     X2 = xi[1]
#     w1 = wxi[0]
#     w2 = wxi[1]
#     X, wx = couple(X1, X2, w1, w2)
    
#     marginals_list = [X] + yi
#     inhomogeneous_xy = list(itertools.product(*marginals_list))
#     xy_set = np.vstack([np.hstack(inhomogeneous_xy[t]) for t in range(len(inhomogeneous_xy))])
    
#     w_list = [wx] + wyi
#     inhomogeneous_w = list(itertools.product(*w_list))
#     w = np.vstack([np.hstack(inhomogeneous_w[t]) for t in range(len(inhomogeneous_w))])
#     w = w.prod(axis=1)
    
#     return xy_set, w
    
# def update_weight(working_sample, new_weight):
#     working_sample[:, -1] = new_weight
#     return working_sample

# utils - 2d plot
def plot_sample_2d(sample, label='sample', w=None, random_sample_size=1000):
    figsize = [10,10]
    if not w is None:
        selection = np.random.choice(range(len(sample)), size=random_sample_size, p=w)
        sample = sample[selection]
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
# note: the comented pieces includes many different tentative ways to shape the graph (they may be ignored)
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
    figsize = [10,10]
    fig = pl.figure(figsize=figsize)
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
