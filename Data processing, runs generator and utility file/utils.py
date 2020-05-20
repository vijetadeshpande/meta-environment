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
        try:
            filepath = os.path.join(dir_path, filename)
            filename, _ = os.path.splitext(filename)
            data[filename] = load_json(filepath)
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
    tp[target_time:] = 0
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
        x[1:, :, feature] = np.multiply(x[1:, :, feature], sd[feature]) + mean[feature]
    
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
    incidence = expand_incidence(samples['HIVmthIncidMale'][example_n], 200, 40)
    
    # incidence after 100 is 0
    incidence[101:] = 0
    
    # define truncnorm distribution and take samples from it for starting age
    mean, sd = np.divide(samples['InitAge'][example_n][0], 12), np.divide(samples['InitAge'][example_n][1], 12)
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
    
    FEATURE_VEC = ['UseHIVIncidReduction', 'HIVIncidReductionCoefficient', 'HIVmthIncidMale', 'PrEPEnable', 'PrEPCoverage', 'PrepIncidMale']
    
    return FEATURE_VEC

def featurize_in_file(samples, sample_bounds, example_n, SEQ_LEN):
    
    # for simplicity I am a state of the system as a vector in following feature space
    # and order is mainted in such fashion to resemble the CEPAC flow of work
    FEATURE_VEC = get_feature_vector()

    #
    feature_mat = pd.DataFrame(0, index = np.arange(SEQ_LEN), columns = FEATURE_VEC)
    for var in FEATURE_VEC:
        
        # switch like features
        if var in ['UseHIVIncidReduction', 'PrEPEnable', 'HIVIncidReductionCoefficient']:
            val = expand_input(samples[var][example_n], SEQ_LEN)
            if (var == 'HIVIncidReductionCoefficient'):
                if (samples['HIVIncidReductionStopTime'][example_n] < SEQ_LEN-1):
                    # after reduction_stop_time, the factor changes to max value
                    val[samples['HIVIncidReductionStopTime'][example_n]:] = sample_bounds['HIVIncidReductionStopTime']['ub']
                # calculate z-score of the val
                val = z_score(val, sample_bounds['HIVIncidReductionCoefficient']['mean'], sample_bounds['HIVIncidReductionCoefficient']['sd'])
        
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
            
        elif var == 'PrepIncidMale':
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

