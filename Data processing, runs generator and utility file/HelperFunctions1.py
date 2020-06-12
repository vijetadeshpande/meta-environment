#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:53:45 2020

@author: vijetadeshpande
"""

import scipy as sp
from copy import deepcopy
import os
import numpy as np
import _pickle as pickle
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import sys
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/CEPAC-extraction-tool')
import link_to_cepac_in_and_out_files as link
import TextFileOperations as t_op
import cluster_operations as c_op

IGNORE_LIST = [r'.DS_Store']

def dump_json(data, filename):
    with open(filename, 'w') as fp:
        json.dump(data, fp)
    
    return

def load_json(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
        
    return data

def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data

def load_all_json(dir_path):
    
    # list of files that we need to upload
    file_list = os.listdir(dir_path)
    data = {}
    
    # iterate over files
    for filename in file_list:
        if (filename in IGNORE_LIST) or ('.csv' in filename):
            continue
        try:
            filepath = os.path.join(dir_path, filename)
            filename, _ = os.path.splitext(filename)
            data[filename] = load_json(filepath)
        except:
            continue
    
    return data

def load_all_csv(dir_path):
    
    # list of files that we need to upload
    file_list = os.listdir(dir_path)
    data = {}
    
    # iterate over files
    for filename in file_list:
        if (filename in IGNORE_LIST) or ('.json' in filename):
            continue
        try:
            filepath = os.path.join(dir_path, filename)
            #filename, _ = os.path.splitext(filename)
            data[filename] = pd.read_csv(filepath)
        except:
            continue
    
    return data

# function for Z-standardization
def z_score(val, mean, sd):
    # calculate z score
    z_score = np.divide((val - mean), sd)
    
    return z_score

# rate to monthly probability
def rate_to_prob(rate, factor = 1):
    rate = rate/factor
    prob = 1 - np.exp(-rate)
    
    return prob

# weibull distribution for prep uptake probabilities
def weibull_tp(coverage, target_time, shape, horizon):
    t_step = np.arange(horizon)
    t_next = t_step + 1
    scale = -np.log(1 - coverage)/(np.power(target_time, shape))
    tp = 1 - np.exp(np.multiply(scale, np.power(t_step, shape, dtype = float)) - np.multiply(scale, np.power(t_next, shape, dtype = float)))
    
    # modify tp: after target time we don't enroll anyone in prep program
    tp[int(target_time):] = 0
    scale1 = np.divide(target_time, (np.power((-np.log(1 - coverage)), (1/shape))))
    cdf_c = 1 - np.exp(-np.power(np.divide(t_step, scale1, dtype = float), shape, dtype = float))
    
    return tp

# expand inputs to horizon length
def expand_input(val, horizon):
    
    val = np.multiply(val, np.ones((horizon, )))
    
    return val

# de-normalize the output values
def denormalize(x, mean, sd):
    # shape
    SEQ_LEN, BATCH, FEATURES = x.shape
    # loop over features
    for feature in range(0, FEATURES):
        x[1:, :, feature] = (x[1:, :, feature] * sd[feature]) + mean[feature]
        #x[1:, :, feature] = np.multiply(x[1:, :, feature], sd[feature]) + mean[feature]
    
    return x

# incidence calculator for the input feature
def gauss(x, a, b, c):
    
    # gauss functional form
    y = np.multiply(a, np.exp(-1 * np.divide(np.power((x-b), 2), np.power(2*c, 2))))
    
    return y

def gaussian_fitter(val, p_init = [1, 28, 5]):
    
    # age bins as defined in CEPAC and corresponding incidence rate values
    x_d = np.array([16, 18, 25, 30, 40, 46, 51, 55]) # this doesn't chnage
    y_d = np.array([val, val, val, val, 1, 1, 1, 1])
    
    # regress a Gauss curve ove the value
    solution = sp.optimize.curve_fit(gauss, x_d, y_d, p0 = p_init)
    
    # extract optimal values of parameters
    opt_a, opt_b, opt_c = solution[0][0], solution[0][1], solution[0][2]
    
    # get incidence values for the opyimal functional form
    y_hat = gauss(x_d, opt_a, opt_b, opt_c)    
    
    return y_hat, solution

def robbins_monroe(approximation, sample, sample_number):
    
    # stochastic approximation
    approximation = np.multiply((1 - 1/sample_number), approximation) + np.multiply((1/sample_number), sample)
    
    return approximation

def expand_incidence(val, length, step):
    
    #
    incidence = np.zeros(length)
    
    #
    incidence[:step] = val
    incidence[step:] = 1
    
    return incidence

def get_incidence_sequence(samples, example_n, SEQ_LEN):
    
    # TODO: Gaussian fit is too expensive
    # get incidence values for each age from Gaussian fit
    #a0, b0, c0 = samples['Gaussian solution'][example_n][0][0], samples['Gaussian solution'][example_n][0][1], samples['Gaussian solution'][example_n][0][2]
    #incidence = gauss(np.arange(200), a0, b0, c0)
    
    # here we need incidence for each age
    try:
        incidence = expand_incidence(samples['HIVmthIncidMale'][example_n], 200, 40)
    except:
        incidence = expand_incidence(samples['HIVmthIncidMale'], 200, 40)
    
    # incidence after 100 is 0
    incidence[101:] = 0
    
    # define truncnorm distribution and take samples from it for starting age
    try:
        mean, sd = np.divide(samples['InitAge'][example_n][0], 12), np.divide(samples['InitAge'][example_n][1], 12)
    except:
        mean, sd = np.divide(samples['InitAge mean'], 12), np.divide(samples['InitAge sd'], 12)
    lb, ub = 16, 100
    a, b = (lb - mean)/sd, (ub - mean)/sd
    distribution = sp.stats.truncnorm(a, b, loc = mean, scale = sd)
    age_samples = distribution.rvs(1000).astype(int)
    
    # loop over samples and calculate mean incidence
    approximation = np.zeros(SEQ_LEN)
    sample_number = 0
    for start_age in age_samples:
        # update sample number
        sample_number += 1
        
        # what's stop age?
        stop_age = start_age + SEQ_LEN
        
        # current sample of incidence values
        incidence_sample = incidence[start_age : stop_age]
        
        # calculate average
        approximation = robbins_monroe(approximation, incidence_sample, sample_number)
        
    # convert to probability
    approximation = rate_to_prob(approximation, factor = 1200)
    
    return approximation

def get_feature_vector():
    
    FEATURE_VEC = ['UseHIVIncidReduction', 'HIVIncidReductionCoefficient', 'HIVmthIncidMale', 'PrEPEnable', 'PrEPCoverage', 'PrepIncidMale', 'DynamicTransmissionNumTransmissionsHRG', 'DynamicTransmissionPropHRGAttrib', 'DynamicTransmissionNumHIVPosHRG', 'prevalence']
    
    return FEATURE_VEC

def featurize_in_file(samples, sample_bounds, example_n, SEQ_LEN):
    
    
    # for simplicity I am a state of the system as a vector in following feature space
    # and order is mainted in such fashion to resemble the CEPAC flow of work
    FEATURE_VEC = get_feature_vector()

    #
    feature_mat = pd.DataFrame(0, index = np.arange(SEQ_LEN), columns = FEATURE_VEC)
    for var in FEATURE_VEC:
        
        # switch like features (or the features which will have same value for SEQ_LEN)
        if var in ['UseHIVIncidReduction', 'PrEPEnable', 'HIVIncidReductionCoefficient', 'DynamicTransmissionNumTransmissionsHRG', 'DynamicTransmissionPropHRGAttrib', 'DynamicTransmissionNumHIVPosHRG', 'prevalence']:
            val = expand_input(samples[var][example_n], SEQ_LEN)
            if (var == 'HIVIncidReductionCoefficient'):
                if (samples['HIVIncidReductionStopTime'][example_n] < SEQ_LEN-1):
                    # after reduction_stop_time, the factor changes to max value
                    val[samples['HIVIncidReductionStopTime'][example_n]:] = sample_bounds['HIVIncidReductionStopTime']['ub']
                # calculate z-score of the val
                val = z_score(val, sample_bounds['HIVIncidReductionCoefficient']['mean'], sample_bounds['HIVIncidReductionCoefficient']['sd'])
            elif var in ['DynamicTransmissionNumTransmissionsHRG', 'DynamicTransmissionNumHIVPosHRG']:
                # calculate z-score
                val = z_score(val, sample_bounds[var]['mean'], sample_bounds[var]['sd'])
        
            
        # weibull uptake probabilities
        elif var == 'PrEPCoverage':
            target_uptake, target_time, shape = samples['PrEPCoverage'][example_n], samples['PrEPDuration'][example_n], samples['PrEPShape'][example_n]
            val = weibull_tp(target_uptake, target_time, shape, SEQ_LEN)
        
        # generating sequence of the incidence 
        elif var == 'HIVmthIncidMale':
            # calculate sequence of raw incidence
            val = get_incidence_sequence(samples, example_n, SEQ_LEN)
            
            # TODO: following line is hard coded (need to take reference for eff and adhe)
            factor = 1 - np.multiply(0.96, 0.739)
            
            # calculate sequence of prep incidence
            val_prep = np.multiply(factor, val)
            feature_mat.loc[:, 'PrepIncidMale'] = val_prep
            
        elif var in ['PrepIncidMale', 'DynamicTransmissionNumHIVNegHRG']:
            continue
        
        # store
        feature_mat.loc[:, var] = val
    
    return feature_mat.values.tolist()

def tensor_shuffler(data, device):
    
    #
    import torch
    
    # shape 
    BATCH_SIZE, SRC_SEQ, INPUT_DIM = data[0][0].shape
    _, TRG_SEQ, OUTPUT_DIM = data[0][1].shape
    
    #
    x, y = [], []
    for batch in data:
        x += batch[0].numpy().tolist()
        y += batch[1].numpy().tolist()
    
    # shuffle
    data_list = list(zip(x, y))
    random.shuffle(data_list)
    x, y = zip(*data_list)
    x, y = np.array(x), np.array(y)
    
    # create batches
    data_s = []
    for example in range(0, len(x), BATCH_SIZE):
        x_batch, y_batch = x[example:example + BATCH_SIZE, :, :], y[example:example + BATCH_SIZE, :]
        x_batch, y_batch = torch.tensor(x_batch, device = device).type('torch.FloatTensor'), torch.tensor(y_batch, device = device).type('torch.FloatTensor')
        data_s.append((x_batch, y_batch))
    
    return data_s

# Function to insert row in the dataframe 
def Insert_row_(row_number, df, row_value): 
    # Slice the upper half of the dataframe 
    df1 = df[0:row_number] 
   
    # Store the result of lower half of the dataframe 
    df2 = df[row_number:] 
   
    # Inser the row in the upper half dataframe 
    df1.loc[row_number]=row_value 
   
    # Concat the two dataframes 
    df_result = pd.concat([df1, df2]) 
   
    # Reassign the index labels 
    df_result.index = [*range(df_result.shape[0])] 
   
    # Return the updated dataframe 
    return df_result

# when we import all the .out files, it is in form of dictionary.
# this dictionary is sorted in weird way if we use sort(). This results in 
# mismatch of X-Y. Following function sorts the output dictionary according to the 
# digit in the key 
def sort_output_dict(dict_in):
    dict_out = {}
    for file in dict_in:
        _, key = file.split('_')
        key = int(key)
        dict_out[key] = dict_in[file]
    
    # sort
    #dict_out = sorted(dict_out)
    
    return dict_out

# new fuction to change the feature representioation
def new_feature_representation(data_in, data_type = 'list'):
    
    #
    X, Y = np.array(data_in[0][0]), data_in[0][1]
    
    #
    FEATURE_VEC = get_feature_vector()
    EXAMPLES, SRC_LEN, INPUT_DIM = X.shape
    INPUT_DIM_NEW = INPUT_DIM-2
    
    #
    X_new = np.zeros((EXAMPLES, SRC_LEN, INPUT_DIM_NEW))
    
    #
    idx, idx_new = -1, -1
    for feature in FEATURE_VEC:
        idx += 1
        idx_new += 1
        
        if feature == 'PrEPEnable':
            val = np.multiply(X[:, :, idx], X[:, :, idx+1])

        elif feature == 'UseHIVIncidReduction':
            val = np.multiply(X[:, :, idx], X[:, :, idx+1])
            idx_mat = np.logical_and((X[:, :, idx+1] != 0), (val == 0))
            val[idx_mat] = 1.4285714285714286
            
        elif feature in ['HIVIncidReductionCoefficient', 'PrEPCoverage']:
            idx_new -= 1
            continue
        
        else:
            val = X[:, :, idx]
        
        #
        X_new[:, :, idx_new] = val
    
    
    #
    if data_type == 'torch':
        import torch
        X_new = torch.tensor(X_new.tolist()).type('torch.FloatTensor')
        Y = torch.tensor(Y).type('torch.FloatTensor')
    else:
        X_new = X_new.tolist()
    
    #
    data_out = [[X_new, Y]]
            
            
    return data_out
    
# function to convert a cepac '.in' file to a condensed representation of file
def condense_in_file(in_file, var_list, var_loc = {}):
    
    condensed_file = pd.DataFrame(-10, index = [0], columns = var_list)
    
    # iterate over each variable
    for var in var_list:
        
        # get value of variable
        if not var in ['InitAge mean', 'InitAge sd', 'onART', 'prevalence']:
            val = t_op.read_values(var, in_file, position = var_loc[var])
        
            # store value in dataframe
            if var == 'InitAge':
                condensed_file.loc[0, 'InitAge mean'] = val[0]
                condensed_file.loc[0, 'InitAge sd'] = val[1]
            elif var in ['HIVmthIncidMale', 'PrepIncidMale']:
                val = val[0]
                val = -1200 * np.log(1 - val)
                condensed_file.loc[0, var] = val
            elif var in ['PrEPCoverage', 'PrEPDuration', 'PrEPShape']:
                val = val[0]
                condensed_file.loc[0, var] = val
            else:
                condensed_file.loc[0, var] = val
    
    return condensed_file.values[0]


