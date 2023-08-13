# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 11:30:45 2023

@author: souzam
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from cycler import cycler

# plot marginals from xls file
_path = './data/Marginal plot.xlsx'
xl = pd.ExcelFile(_path)
marginals = xl.parse('Marginals', index_col='Strike')

# normalize
marginals = np.maximum(marginals, np.zeros(marginals.shape))
marginals = marginals * .2 / marginals.sum()

p0_AAPL = 134.51  # https://finance.yahoo.com/quote/AAPL/history?period1=1669852800&period2=1672444800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true
p0_AMZN =  87.86  # https://finance.yahoo.com/quote/AMZN/history?period1=1669852800&period2=1672444800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true

amzn = marginals[['AMZN_1', 'AMZN_2']]
amzn = amzn.dropna()
amzn.index = -1.0 + amzn.index / p0_AMZN

aapl = marginals[['AAPL_1', 'AAPL_2']]
aapl = aapl.dropna()
aapl.index = -1.0 + aapl.index / p0_AAPL

# chosen color cycler (see empirical example)
cc = cycler('color', ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00', '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2'])

# plot
# fig, ax = pl.subplots(1, 2, figsize = [12,4], sharex=True, sharey=True)   # plot in two iterations to have a clean legend
# [a.set_prop_cycle(cc) for a in ax]
# ax[0].plot(marginals[['AMZN_1', 'AMZN_2']])
# ax[0].fill_between(marginals.index, marginals['AMZN_1'], alpha = .25)
# ax[0].fill_between(marginals.index, marginals['AMZN_2'], alpha = .25)
# ax[0].set_xlabel('Amazon')
# ax[1].plot(marginals[['AAPL_1', 'AAPL_2']])
# ax[1].fill_between(marginals.index, marginals['AAPL_1'], alpha = .25)
# ax[1].fill_between(marginals.index, marginals['AAPL_2'], alpha = .25)
# ax[1].set_xlabel('Apple')
# ax[1].legend(['Jan. 20th, 2023', 'Feb. 17th, 2023'], loc='upper left')
# fig.tight_layout()
# fig.show()

fig, ax = pl.subplots(1, 2, figsize = [9,4], sharex=True, sharey=True)   # plot in two iterations to have a clean legend
[a.set_prop_cycle(cc) for a in ax]
ax[0].plot(amzn)
ax[0].fill_between(amzn.index, amzn['AMZN_1'], alpha = .25)
ax[0].fill_between(amzn.index, amzn['AMZN_2'], alpha = .25)
ax[0].set_xlabel('Amazon')
ax[1].plot(aapl)
ax[1].fill_between(aapl.index, aapl['AAPL_1'], alpha = .25)
ax[1].fill_between(aapl.index, aapl['AAPL_2'], alpha = .25)
ax[1].set_xlabel('Apple')
ax[1].legend(['Jan. 20th, 2023', 'Feb. 17th, 2023'], loc='upper left')
fig.tight_layout()
fig.show()


