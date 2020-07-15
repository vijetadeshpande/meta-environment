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
import HelperFunctions1 as h_fun1
import HelperFunctions2 as h_fun2
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
PATH_RAW = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/CEPAC RUNS'#r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/CEPAC RUNS'
IGNORE = [r'.DS_Store', r'CURRENT BATCH OF CEPAC RUNS', r'10 correct SQ runs', 'regression model input', 'NEW BATCH_1', 'NEW BATCH_2']#, 'NEW BATCH', 'this']
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
        data_rnn[batch] = h_fun1.load_all_json(path_rnn)
        
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
            h_fun1.dump_json(data_cepac[batch], filename)
        
        else:
            data_cepac[batch] = data_rnn[batch].pop('CEPAC_output')
            
# coverting keys from str to int
for batch in data_cepac:
    data_cepac[batch] = h_fun1.sort_output_dict(data_cepac[batch])

# now we need all the data in 'data_cepac' in one place to calculate mean, sd anf standardize the output data from it
TOTAL_EXAMPLES = sum([len(data_cepac[i]) for i in data_cepac])
VARIABLES = len(data_rnn[batch]['CEPAC_input'])
cepac_input, cepac_output =  np.zeros((TOTAL_EXAMPLES, VARIABLES)), np.zeros((TOTAL_EXAMPLES, SEQ_LEN+1, OUTPUT_DIM))
rnn_input, rnn_output = np.zeros((TOTAL_EXAMPLES, SEQ_LEN, INPUT_DIM)), np.zeros((TOTAL_EXAMPLES, SEQ_LEN+1, OUTPUT_DIM))

#idx_example = -1
for batch in data_cepac:
    #
    if batch in IGNORE:
        continue
    
    # what is batch length
    # TODO: following two lines works on the assumption that the batch size
    # is constant, always. And is equal to 1000
    BATCH_SIZE = pd.DataFrame(data_rnn[batch]['CEPAC_input']).shape[0]
    START_INDEX, END_INDEX = int(batch)*1000, int(batch)*1000 + BATCH_SIZE
    
    # check if we have cepac input tensor and collect 
    if not 'CEPAC_input' in data_rnn[batch]:
        # get list of variables and dictionary to store values 
        samples, sample_bounds, var_list, parameters = ipar.get_samples(1)
        
        #
        cepac_batch_input = pd.DataFrame(-10, index = np.arange(len(data_cepac[batch])), columns = var_list)
        
        # read all cepac .in files
        CEPAC_input_files = link.import_all_cepac_in_files(os.path.join(PATH_RAW, batch, 'Files for CEPAC'))
        
        # save location of each variable
        var_loc = {}
        for file in CEPAC_input_files:
            for var in var_list:
                if not var in var_loc:
                    try:
                        loc = t_op.search_var(var, CEPAC_input_files[file])
                    except:
                        loc = {}
                    # store location of the variable
                    var_loc[var] = loc
            break
        
        # store value of each sample
        for file in CEPAC_input_files:
            _, idx_file = file.split('_')
            idx_file = int(idx_file) #+ (int(batch)*1000 - 5000)
            cepac_batch_input.iloc[idx_file, :] = h_fun1.condense_in_file(CEPAC_input_files[file], var_list, var_loc)
    
        # add to cepac_input
        cepac_input[START_INDEX:END_INDEX, :, :] = cepac_batch_input.values#.tolist()
        
        # save condensed cepac input files
        cepac_batch_input = cepac_batch_input.to_dict()
        h_fun1.dump_json(cepac_batch_input, os.path.join(os.path.join(PATH_RAW, batch, 'Files for RNN'), 'CEPAC_input.json'))
        
        # store
        data_rnn[batch]['CEPAC_input'] = cepac_batch_input
        
    else:
        cepac_input[START_INDEX:END_INDEX, :] = pd.DataFrame(data_rnn[batch]['CEPAC_input']).values#.tolist()
    
    
    # check if we have RNN input tensor and collect
    if not 'RNN_source' in data_rnn[batch]:
        
        # define condensed input and input bounds (required for standardization)
        cepac_batch_input = pd.DataFrame(data_rnn[batch]['CEPAC_input'])
        input_bounds = h_fun1.load_json('input_bounds.json')
        
        rnn_batch_input = []
        for file in cepac_batch_input.index:
            x = h_fun2.build_state(cepac_batch_input.loc[file, :], input_bounds, output_type = 'list')
            rnn_batch_input.append(x)
        
        # store
        data_rnn[batch]['RNN_source'] = rnn_batch_input
        
        # save
        h_fun1.dump_json(rnn_batch_input, os.path.join(os.path.join(PATH_RAW, batch, 'Files for RNN'), 'RNN_source.json'))
        
        # collect
        rnn_input[START_INDEX:END_INDEX, :, :] = data_rnn[batch]['RNN_source']
        
    else:
        # collect
        rnn_input[START_INDEX:END_INDEX, :, :] = np.array(data_rnn[batch]['RNN_source'])
        
        
    # get CEPAC output and collect
    cepac_batch_output = np.zeros((len(data_cepac[batch]), SEQ_LEN+1, OUTPUT_DIM))
    for example in data_cepac[batch]:
        idx_example = example #(1000 * int(batch)) + example
        cepac_batch_output[idx_example, 1:, :] = pd.DataFrame(data_cepac[batch][example])
    cepac_output[START_INDEX:END_INDEX, :, :] = cepac_batch_output#.tolist()
    
# calculate mean and std of output values
mean, std = np.zeros((OUTPUT_DIM)), np.zeros((OUTPUT_DIM))
for feature in range(OUTPUT_DIM):
    # find mean and sd
    mean[feature] = np.mean(cepac_output[:, :, feature])
    std[feature] = np.std(cepac_output[:, :, feature])
    
    # standardize the data
    rnn_output[:, 1:, feature] = (cepac_output[:, 1:, feature] - mean[feature])/std[feature]
    

# few adjustments
rnn_input, cepac_input = rnn_input.tolist(), cepac_input.tolist()
rnn_output, cepac_output = rnn_output.tolist(), cepac_output.tolist()


# convert into pandas dataframe
X, Y = pd.DataFrame(rnn_input, index = np.arange(TOTAL_EXAMPLES)), pd.DataFrame(rnn_output, index = np.arange(TOTAL_EXAMPLES))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.25)

# extract and save labels of the runs for train and test datasets
labels_train, labels_test, labels_val = X_train.index, X_test.index, X_val.index
labels = pd.DataFrame(-1, index = np.arange(len(labels_train)), columns = ['train', 'test', 'validation'])
labels.loc[:, 'train'], labels.loc[0:len(labels_test)-1, 'test'], labels.loc[0:len(labels_val)-1, 'validation'] = labels_train, labels_test, labels_val

#%% SAVE FILES

# train  and test labels
labels.to_csv(os.path.join(path_regression_in, 'labels.csv'))

# pickle the data
h_fun1.dump_json([(X_train.values.tolist(), Y_train.values.tolist())], os.path.join(path_regression_in, 'train.json'))
h_fun1.dump_json([(X_test.values.tolist(), Y_test.values.tolist())], os.path.join(path_regression_in, 'test.json'))
h_fun1.dump_json([(X_val.values.tolist(), Y_val.values.tolist())], os.path.join(path_regression_in, 'validation.json'))


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
h_fun1.dump_json([(cepac_input, cepac_output)], os.path.join(path_regression_in, 'CEPAC_data_unshuffled.json'))
h_fun1.dump_json([(rnn_input, rnn_output)], os.path.join(path_regression_in, 'RNN_data_unshuffled.json'))
