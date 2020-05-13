#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:00:37 2020

@author: vijetadeshpande
"""

import scipy as sp
from copy import deepcopy
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from copy import deepcopy
import timeit
import input_parameters as ipar
import utils
import sys
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/CEPAC-extraction-tool')
import link_to_cepac_in_and_out_files as link
import TextFileOperations as t_op
import cluster_operations as c_op


#%% RAW DATA COLLECTION
# 1. get list of directories which have the raw data 
# 2. import all data (if there's no compressed file, else import compressed data)
# 3. save the data (for each directory and all data combined)
PATH_RAW = r'/Users/vijetadeshpande/Documents/GitHub/Sequence2Sequence model for CEPAC prediction/Data and results/CEPAC RUNS'
IGNORE = [r'.DS_Store', r'CURRENT BATCH OF CEPAC RUNS', r'10 correct SQ runs', 'regression model input']
SEQ_LEN = 60
batch_list = os.listdir(PATH_RAW)

# create a directory to save the input data for model
path_regression_in = os.path.join(PATH_RAW, 'regression model input')
if not os.path.exists(path_regression_in):
    os.makedirs(path_regression_in)

# iterate over list of bath paths
data_raw = {}
data_comp = {}
for batch in batch_list:    
    if batch in IGNORE:
        continue
    
    # important paths
    path_batch = os.path.join(PATH_RAW, batch)
    path_comp = os.path.join(path_batch, r'COMPRESSED FILES')
    path_input = os.path.join(path_batch, r'INPUT FILES')
    path_output = os.path.join(path_batch, r'OUTPUT FILES')
    
    # initiate a dictionary
    data_comp[batch] = {}
    
    #
    if os.path.exists(os.path.join(path_comp)):
        # load data
        data_comp[batch] = utils.load_all_json(path_comp)
    else:
        # first make the directory
        os.makedirs(path_comp)
        
        # collect the output which was parallelized for the cluster
        if os.path.exists(os.path.join(path_input, str(0))):
            c_op.collect_output(path_input)
        
        # import raw data
        data_raw[batch] = link.import_all_cepac_out_files(os.path.join(path_input, 'results'), module = 'regression')
        
        # collect the raw data by each feature
        seq_out_by_features = {'transmissions': [], 'infections': [], 'susceptibles': []}
        for ex in data_raw[batch]:
            for feat in data_raw[batch][ex]:
                if feat == 'multiplier':
                    continue
                elif feat == 'susceptibles':
                    seq_out_by_features[feat].append(data_raw[batch][ex][feat].values[:SEQ_LEN].tolist())
                else:
                    seq_out_by_features[feat].append(data_raw[batch][ex][feat].values[:SEQ_LEN].tolist())

        # save raw data to json file
        data_comp[batch]['output raw'] = seq_out_by_features
        filename = os.path.join(path_comp, r'output raw.json')
        utils.dump_json(seq_out_by_features, filename)
        
    # import standardized input data
    data_comp[batch]['input std'] = utils.load_jsonl(os.path.join(path_output, 'input_tensor.json'))
    
    # destandardization of the input data
    # ?

# we have all the raw output data that we need for standardization
collected_data_out = {'transmissions': [], 'infections': [], 'susceptibles': []}
collected_data_in = []
for file in data_comp:
    for out in data_comp[file]:
        if out == 'output raw':
            for feat in data_comp[file][out]:
                collected_data_out[feat] = collected_data_out[feat] + data_comp[file][out][feat]   
        elif out == 'input std':
            collected_data_in = collected_data_in + data_comp[file][out]
            

recom_mean, recom_sd = {'transmissions': 0, 'infections': 0, 'susceptibles': 0}, {'transmissions': 0, 'infections': 0, 'susceptibles': 0}
for feat in collected_data_out:
    recom_mean[feat], recom_sd[feat] = np.mean(collected_data_out[feat]), np.std(collected_data_out[feat])
# standardize
EXAMPLES = len(collected_data_out[feat])

#%% WHICH OUTPUT FEATURES WE WANT TO USE FOR REGRESSION

OUT_FEATURES = ['infections', 'susceptibles'] #['transmissions', 'infections', 'susceptibles']

#%%
output_tensor = np.zeros((EXAMPLES, SEQ_LEN+1, len(OUT_FEATURES)))
feat_idx = -1
for feat in OUT_FEATURES:
    feat_idx += 1
    output_tensor[:, 1:, feat_idx] = utils.z_score(collected_data_out[feat], 
                                                 recom_mean[feat],
                                                 recom_sd[feat])

#
IN_FEATURES = len(collected_data_in[0][0])
input_tensor = np.zeros((EXAMPLES, SEQ_LEN, IN_FEATURES))
input_tensor[:, :, :] = collected_data_in

# save standardized input and output tensor for regression model
# split the data
X = []
Y = []
for example in range(0, EXAMPLES):
    X.append(input_tensor[example, :, :].tolist())
for example in range(0, EXAMPLES):
    Y.append(output_tensor[example, :, :].tolist())

# convert into pandas dataframe
X, Y = pd.DataFrame(X, index = np.arange(EXAMPLES)), pd.DataFrame(Y, index = np.arange(EXAMPLES))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# extract and save labels of the runs for train and test datasets
labels_train, labels_test = X_train.index, X_test.index
labels = pd.DataFrame(-1, index = np.arange(len(labels_train)), columns = ['train', 'test'])
labels.loc[:, 'train'], labels.loc[0:len(labels_test)-1, 'test'] = labels_train, labels_test
labels.to_csv(os.path.join(path_regression_in, 'labels.csv'))

# pickle the data
utils.dump_json([(X_train.values.tolist(), Y_train.values.tolist())], os.path.join(path_regression_in, 'train.json'))
utils.dump_json([(X_test.values.tolist(), Y_test.values.tolist())], os.path.join(path_regression_in, 'test.json'))

# save the mean and sd of the output data
mean_sd = pd.DataFrame(0, index = ['mean', 'sd'], columns = OUT_FEATURES)
for feat in OUT_FEATURES:
    mean_sd.loc['mean', feat] = recom_mean[feat]
    mean_sd.loc['sd', feat] = recom_sd[feat]
mean_sd.to_csv(os.path.join(path_regression_in, r'output_mean_and_sd.csv'))
