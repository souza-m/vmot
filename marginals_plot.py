# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 11:30:45 2023

@author: souzam
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

# plot marginals from xls file
_path = './data/Marginal plot.xlsx'
xl = pd.ExcelFile(_path)
marginals = xl.parse('Marginals', index_col='Strike')

# normalize
marginals = np.maximum(marginals, np.zeros(marginals.shape))
marginals = marginals * .2 / marginals.sum()


# plot
fig, ax = pl.subplots(1, 2, figsize = [9,4], sharey=True)   # plot in two iterations to have a clean legend
ax[0].plot(marginals[['AMZN_1', 'AMZN_2']])
ax[0].fill_between(marginals.index, marginals['AMZN_1'], alpha = .25)
ax[0].fill_between(marginals.index, marginals['AMZN_2'], alpha = .25)
ax[0].set_xlabel('Amazon')
ax[1].plot(marginals[['AAPL_1', 'AAPL_2']])
ax[1].fill_between(marginals.index, marginals['AAPL_1'], alpha = .25)
ax[1].fill_between(marginals.index, marginals['AAPL_2'], alpha = .25)
ax[1].set_xlabel('Apple')
ax[1].legend(['Jan. 20th, 2023', 'Feb. 17th, 2023'], loc='upper right')
fig.tight_layout()
fig.show()

