#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 17:08:11 2020

@author: vijetadeshpande
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/Sequence2Sequence model for CEPAC prediction/Data processing, runs generator and utility file')
import HelperFunctions1 as h_fun1


# import data for the error plots
error = h_fun1.load_all_json(r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Hyper parameter tuning/trained model all results')
EPOCHS, N_MODELS, ERROR_TYPE = 100, 3, 2
plot_df = pd.DataFrame(0, index = np.arange(EPOCHS * N_MODELS * ERROR_TYPE), columns = ['Epoch', 'MSE', 'Error type', 'Model'])

idx = 0
for model in error:
    if 'GRU' in model:
        plot_df.loc[idx:idx + (EPOCHS * ERROR_TYPE), 'Model'] = 'RNN GRU'
    elif 'LSTM' in model:
        plot_df.loc[idx:idx + (EPOCHS * ERROR_TYPE), 'Model'] = 'RNN LSTM'
    elif 'Van' in model:
        plot_df.loc[idx:idx + (EPOCHS * ERROR_TYPE), 'Model'] = 'RNN Vanilla'
        
    #
    plot_df.loc[idx:idx + (EPOCHS * ERROR_TYPE)-1, 'Epoch'] = np.arange(100).tolist() * ERROR_TYPE
    plot_df.loc[idx:idx + (EPOCHS * ERROR_TYPE)-1, 'Error type'] = ['Training'] * EPOCHS + ['Validation'] * EPOCHS
    plot_df.loc[idx:idx + (EPOCHS * ERROR_TYPE)-1, 'MSE'] = error[model]['1']['results']['train loss'] + error[model]['1']['results']['validation loss']
    
    #
    idx += (EPOCHS * ERROR_TYPE)
    
# plot
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set_context("notebook", rc={"lines.linewidth": 1.5})#, font_scale = 1.2)
plt.figure()
g = sns.FacetGrid(data = plot_df, 
                 col = 'Model',
                 sharex = False,
                 sharey = True,
                 aspect = 1)#, hue="Coverage time")#, col_wrap=3)
g = (g.map(sns.lineplot, 
           "Epoch", 
           'MSE',
           'Error type',
           alpha = 1).add_legend())#, "WellID")
plt.savefig('Training and validation error.png', dpi = 360)

    