#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 03:14:14 2020

@author: vijetadeshpande
"""
import sys
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data processing, runs generator and utility file') 
import utils
import os
import numpy as np
import pandas as pd
from copy import deepcopy
import itertools
import torch
from FunctionApproximator import GRUApproximator as FGRU
import utils2
import utils

#%% SOME PAR

DEVICE = 'cpu' 
SRC_LEN = 60
TRG_LEN = 61

# imports
SQ_inputs = pd.read_csv('city_specific_inputs.csv')
SQ_inputs = SQ_inputs.set_index('city')
SQ_inputs['PrEPCoverage'],  SQ_inputs['PrEPDuration']= 0.001, 0
input_bounds = utils.load_json(r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/CEPAC RUNS/regression model input/input_mean_and_sd.json')

#%% STRATEGIES

# what strategies we want to simulate?
uptake = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
uptake_t = [24, 36, 48, 60]
uptake_s = [2]

# create dictionary of input parameters necessary to create an input signal
strategies = list(itertools.product(uptake, uptake_t, uptake_s))
FEATURE_VEC = utils.get_feature_vector()
EXAMPLES, SRC_LEN, INPUT_DIM, OUTPUT_DIM, DEVICE = len(strategies), 60, len(FEATURE_VEC), 3, 'cpu'

# initialize 
Run_A = utils2.build_state(SQ_inputs.loc['rio', :], input_bounds, DEVICE).repeat(EXAMPLES, 1, 1)
Run_B = torch.zeros((EXAMPLES, SRC_LEN, INPUT_DIM)).type('torch.FloatTensor').to(DEVICE)
Y = torch.zeros((EXAMPLES, SRC_LEN+1, OUTPUT_DIM)).type('torch.FloatTensor').to(DEVICE)

# create rnn inputs for RunA and RunB for each strategy
for city in ['rio']:
    idx_strategy = -1
    for (cov, cov_t, shape) in strategies:
        idx_strategy += 1
        
        # replace PrEP values in sq
        INT_float = deepcopy(SQ_inputs)
        INT_float.loc[city, 'PrEPCoverage'], INT_float.loc[city, 'PrEPDuration'], INT_float.loc[city, 'PrEPShape'] = cov, cov_t, shape
        
        # enable PrEP module
        INT_float.loc[:, 'PrEPEnable'] = 1
        
        # convert set of inputs to feature matrix
        Run_B[idx_strategy, :, :] = utils2.build_state(INT_float.loc[city, :], input_bounds, DEVICE)


#%% COMMUNITY BEN

# initialize the model object
filepath = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/RNN GRU/best results/new_features.pt'
z_path = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/CEPAC RUNS/regression model input'
Environment = FGRU(filepath)

# dictionary for storage
ALL_per_red = {}
ALL_red_coe = {}

# forward pass
data = utils.new_feature_representation([[Run_A, Y]], data_type = 'torch')
SQ_ = Environment(data, z_path, DEVICE)
data = utils.new_feature_representation([[Run_B, Y]], data_type = 'torch')
INT_ = Environment(data, z_path, DEVICE)

# iterate over all possible values (should be 24)
for strategy in range(0, len(strategies)):
    
    # TX algo
    percentage_reduction, reduction_coeff = utils2.community_benefit(SQ_['denormalized prediction'][0][:, strategy, :], INT_['denormalized prediction'][0][:, strategy, :])
    
    # save community benefit
    key = str(strategies[strategy])
    ALL_per_red[key], ALL_red_coe[key] = percentage_reduction, reduction_coeff
    
    
# plot heatmap from the saved values
    