#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 20:30:19 2020

@author: vijetadeshpande
"""
import sys
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Combination model')
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data processing, runs generator and utility file')
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Transformer')
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/RNN Vanilla')

#from main_GRU import init_training
#from main_Combination import init_training
#from main_Transformer import init_training
#from main_Vanilla import init_training
from main_LSTM import init_training
import numpy as np
import itertools
from ModelData import ModelData
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import HelperFunctions1 as h_fun1

# path variables
datapath = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/CEPAC RUNS/regression model input'
respath = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/RNN results'

# load data
data_object = ModelData(datapath, batch_size = 16)

# Grid for hyper-par search
hidden_dims = [256]
n_layers = [2]
l_rates = [0.0006]#[0.01, 0.001, 0.00001]#np.power(10, np.random.normal(loc=-3.5, scale=0.7, size=10))
n_epochs = [30]
n_heads = [4]
dropouts = [0]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#
pars = list(itertools.product(hidden_dims, n_layers, dropouts, l_rates, n_epochs, n_heads))
results = {}
idx = 0
for par in pars:
    idx += 1
    
    # print 
    print(('\nStarting training for parameter set: %d')%(idx))
    
    # create dictionary of parameter values
    (hidden_dim, n_layer, dropout, l_rate, n_epoch, n_head) = par
    par_dict = {'hidden dimension': hidden_dim,
                'number of layers': n_layer,
                'dropout rate': dropout,
                'learning rate': l_rate,
                'number of epochs': n_epoch,
                'number of heads': n_head,
                'device': DEVICE
                }
    
    # train with current set of paramters
    results[idx] = {}
    results[idx]['results'] = init_training(data_object, par_dict, datapath, respath)
    _ = par_dict.pop('device')
    results[idx]['hyperparameters'] = par_dict


# plot training error
plot_df = pd.DataFrame(0, index = np.arange(len(pars) * n_epochs[0]), columns = ['Epoch', 'MSE', 'Learning rate'])
idx = 0
idx_par = 0
for result in results:
    idx_par += 1
    plot_df.loc[idx : idx+n_epochs[0]-1, 'Epoch'] = np.arange(n_epochs[0])
    plot_df.loc[idx : idx+n_epochs[0]-1, 'MSE'] = results[result]['results']['train loss']
    plot_df.loc[idx : idx + n_epochs[0]-1, 'Learning rate'] = str(idx_par)
    idx += n_epochs[0]
    
# plot
plot_df['Learning rate'] = plot_df['Learning rate'].astype('str')
plt.figure(figsize=(8, 6))
sns.lineplot(data = plot_df,
             x = 'Epoch',
             y = 'MSE',
             hue = 'Learning rate')
plt.savefig('final error plot2.jpeg', dpi = 360)
plot_df.to_csv('MSEs.csv')
h_fun1.dump_json(results, 'all_results.json')
    
    

    

