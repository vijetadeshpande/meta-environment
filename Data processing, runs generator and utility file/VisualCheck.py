#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 01:21:07 2020

@author: vijetadeshpande
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import sys
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/Sequence2Sequence model for CEPAC prediction/Data processing, runs generator and utility file')
import utils

def get_observations(pair_data, key, filter_labels):
    
    X, Y = pd.DataFrame(data[key][0][0]).iloc[filter_labels, :], pd.DataFrame(data[key][0][1]).iloc[filter_labels, :]
    
    return X, Y

# paths
datapath = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/CEPAC RUNS/regression model input'
labelfile = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/CEPAC RUNS/regression model input/labels.csv'

# import data
data = utils.load_all_json(datapath)
labels = pd.read_csv(labelfile)
test_labels = labels.loc[0:9, 'test'].values

# filter out required data
X_test, Y_test = {}, {}
for key in data:
    if key == 'train':
        continue
    
    if key == 'test':
        label_list = np.arange(10)
    else:
        label_list = test_labels
    X_test[key], Y_test[key] = get_observations(data, key, label_list)

# collect output
FEATURES, SEQ_LEN, N_FILES, N_EXAMPLES = 3, 60, len(X_test), len(test_labels)
total_rows = SEQ_LEN * N_FILES * N_EXAMPLES
test_df = pd.DataFrame(-100, index = np.arange(total_rows), columns = ['t (simulation month)', 'transmissions', 'infections', 'susceptibles', 'files', 'example'])
idx = 0
for file in Y_test:
    float_array = np.reshape(np.array(Y_test[file].to_numpy().tolist()), (N_EXAMPLES*SEQ_LEN, FEATURES))
    start_idx, end_idx = idx, idx+(N_EXAMPLES*SEQ_LEN)
    
    # feature values
    test_df.loc[start_idx: end_idx-1, ['transmissions', 'infections', 'susceptibles']] = float_array
    
    # file
    test_df.loc[start_idx: end_idx-1, 'file'] = file
    
    # example
    test_df.loc[start_idx: end_idx-1, 'example'] = np.reshape(np.array([[i] * SEQ_LEN for i in test_labels]), (N_EXAMPLES*SEQ_LEN, 1))
    
    # simulation time
    test_df.loc[start_idx: end_idx-1, 't (simulation month)'] = [i for i in range(SEQ_LEN)] * N_EXAMPLES
    
    # update idx
    idx = idx+(N_EXAMPLES*SEQ_LEN)


# save plots

# set plot encironment
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set_context("notebook", rc={"lines.linewidth": 1.5}, font_scale = 1.2)
aspect_r = 2
line_alpha = 0.9

# where to save?
figpath = os.path.join(datapath, 'visual check plots')
if not os.path.exists(figpath):
    os.makedirs(figpath)

for example in test_labels:
    
    # plot title
    [enable, cov, cov_t] = X_test['CEPAC_data_unshuffled'].loc[example, [6, 7, 8]]
    plot_title = 'PrEP (En, Co, Co_t) = (%d, %d, %d)'%(enable, cov*100, cov_t)
    
    for feature in ['transmissions', 'infections']:
        
        # filter out required values
        df = test_df.loc[test_df.loc[:, 'example'] == example, :]

        #
        fig = plt.figure()
        g = sns.FacetGrid(data = df, 
                         row = 'file', 
                         sharex = False,
                         sharey = False,
                         aspect = aspect_r)#, hue="Coverage time")#, col_wrap=3)
        g = (g.map(sns.lineplot, 
                   "t (simulation month)", 
                   feature, 
                   alpha = line_alpha))#, "WellID")
        
        # title
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(plot_title)
        
        #
        filename = os.path.join(figpath, r'Output for example = %d and feature = %s'%(example, feature))
        plt.savefig(filename, dpi = 360)
    
#x = r'/Users/vijetadeshpande/Downloads/MPEC/Brazil/Rio/2-way SA_40%/Status quo/results'
#link.export_output_to_excel(x, x)