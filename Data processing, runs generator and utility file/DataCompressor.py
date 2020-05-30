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

"""
x_data = np.array(data_comp['0']['input std'])
y_data = np.array([data_comp['0']['output raw']['transmissions'], data_comp['0']['output raw']['infections']])
y_data = np.transpose(y_data, axes = [1,2,0])

"""

#%% RAW DATA COLLECTION
# 1. get list of directories which have the raw data 
# 2. import all data (if there's no compressed file, else import compressed data)
# 3. save the data (for each directory and all data combined)
PATH_RAW = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/CEPAC RUNS'
IGNORE = [r'.DS_Store', r'CURRENT BATCH OF CEPAC RUNS', r'10 correct SQ runs', 'regression model input', 'NEW BATCH INPUT1', 'NEW BATCH INPUT']#, 'NEW BATCH', 'this']
SEQ_LEN, INPUT_DIM, OUTPUT_DIM = 60, 10, 3
batch_list = os.listdir(PATH_RAW)

# collect output of the new batch
#if os.path.exists(os.path.join(PATH_RAW, 'NEW BATCH', str(0))):
#    c_op.collect_output(os.path.join(PATH_RAW, 'NEW BATCH'))

# create a directory to save the input data for model
path_regression_in = os.path.join(PATH_RAW, 'regression model input')
if not os.path.exists(path_regression_in):
    os.makedirs(path_regression_in)

# iterate over list of bath paths
data_cepac = {}
data_rnn = {}
for batch in batch_list:    
    if batch in IGNORE:
        continue
    
    # important paths
    path_batch = os.path.join(PATH_RAW, batch)
    path_cepac = os.path.join(path_batch, r'Files for CEPAC')
    path_rnn = os.path.join(path_batch, r'Files for RNN')
    
    # initiate a dictionary
    data_rnn[batch] = {}
    
    #
    if os.path.exists(os.path.join(path_rnn)):
        # load data
        data_rnn[batch] = utils.load_all_json(path_rnn)
        
        # check if we have CEPAC output
        #if not 'rnn_target' in data_rnn:
            
        if not 'CEPAC_output' in data_rnn[batch]:
            
            # collect the output which was parallelized for the cluster
            if os.path.exists(os.path.join(path_cepac, str(0))):
                c_op.collect_output(path_cepac)
                
            # import raw data
            data_cepac[batch] = link.import_all_cepac_out_files(os.path.join(path_cepac, 'results'), module = 'regression')
            
            for example in data_cepac[batch]:
                data_cepac[batch][example].pop('multiplier')
                for feature in data_cepac[batch][example]:
                    data_cepac[batch][example][feature] = data_cepac[batch][example][feature].to_list()[:SEQ_LEN]

            # save raw data to json file
            filename = os.path.join(path_rnn, r'CEPAC_output.json')
            utils.dump_json(data_cepac[batch], filename)
        
        else:
            data_cepac[batch] = data_rnn[batch].pop('CEPAC_output')

# now we need all the data in 'data_cepac' in one place to calculate mean, sd anf standardize the output data from it
TOTAL_EXAMPLES = sum([len(data_cepac[i]) for i in data_cepac])
cepac_output, rnn_output = np.zeros((TOTAL_EXAMPLES, SEQ_LEN, OUTPUT_DIM)), np.zeros((TOTAL_EXAMPLES, SEQ_LEN, OUTPUT_DIM))
cepac_input, rnn_input = [], []
idx_example = -1
for batch in data_cepac:
    if batch in IGNORE:
        continue
    cepac_input += pd.DataFrame(data_rnn[batch]['CEPAC_input']).values.tolist()
    rnn_input += data_rnn[batch]['RNN_source']
    for example in data_cepac[batch]:
        idx_example += 1
        idx_feature = -1
        for feature in data_cepac[batch][example]:
            idx_feature += 1
            cepac_output[idx_example, :, idx_feature] = data_cepac[batch][example][feature]
        
# calculate mean and std of output values
mean, std = np.zeros((OUTPUT_DIM)), np.zeros((OUTPUT_DIM))
for feature in range(OUTPUT_DIM):
    # find mean and sd
    mean[feature] = np.mean(cepac_output[:, :, feature])
    std[feature] = np.std(cepac_output[:, :, feature])
    
    # standardize the data
    rnn_output[:, :, feature] = (cepac_output[:, :, feature] - mean[feature])/std[feature]
    

# few adjustments
rnn_output, cepac_output = rnn_output.tolist(), cepac_output.tolist()

# convert into pandas dataframe
X, Y = pd.DataFrame(rnn_input, index = np.arange(TOTAL_EXAMPLES)), pd.DataFrame(rnn_output, index = np.arange(TOTAL_EXAMPLES))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# extract and save labels of the runs for train and test datasets
labels_train, labels_test = X_train.index, X_test.index
labels = pd.DataFrame(-1, index = np.arange(len(labels_train)), columns = ['train', 'test'])
labels.loc[:, 'train'], labels.loc[0:len(labels_test)-1, 'test'] = labels_train, labels_test

#%% SAVE FILES

# train  and test labels
labels.to_csv(os.path.join(path_regression_in, 'labels.csv'))

# pickle the data
utils.dump_json([(X_train.values.tolist(), Y_train.values.tolist())], os.path.join(path_regression_in, 'train.json'))
utils.dump_json([(X_test.values.tolist(), Y_test.values.tolist())], os.path.join(path_regression_in, 'test.json'))

# save the mean and sd of the output data
OUT_FEATURES = ['transmissions', 'infections', 'susceptibles']
mean_sd = pd.DataFrame(0, index = ['mean', 'sd'], columns = OUT_FEATURES)
idx_feature = -1
for feature in OUT_FEATURES:
    idx_feature += 1
    mean_sd.loc['mean', feature] = mean[idx_feature]
    mean_sd.loc['sd', feature] = std[idx_feature]
mean_sd.to_csv(os.path.join(path_regression_in, r'output_mean_and_sd.csv'))

# unshuffled data for reference
utils.dump_json([(cepac_input, cepac_output)], os.path.join(path_regression_in, 'CEPAC_data_unshuffled.json'))
utils.dump_json([(rnn_input, rnn_output)], os.path.join(path_regression_in, 'RNN_data_unshuffled.json'))
