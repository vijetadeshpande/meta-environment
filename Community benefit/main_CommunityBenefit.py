#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 03:14:14 2020

@author: vijetadeshpande
"""
import os
import numpy as np
import pandas as pd
from copy import deepcopy
import itertools
import torch
from FunctionApproximator import GRUApproximator as FGRU
from FunctionApproximator import TransformerApproximator as FTrans
from FunctionApproximator import VanillaApproximator as FVan
import sys
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data processing, runs generator and utility file') 
import HelperFunctions2 as h_fun2
import HelperFunctions1 as h_fun1

#%% SOME PAR

DEVICE = 'cpu' 
SRC_LEN = 60
TRG_LEN = 61
OUT_DIM = 3
TARGET_INT = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/Community benefit CEPAC runs/Measurement of community benefit_mm/Positive coverage runs_mm/results'
TARGET_SQ = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/Community benefit CEPAC runs/Measurement of community benefit_mm/Status quo_mm/results'

# imports
SQ_inputs = pd.read_csv('city_specific_inputs.csv')
SQ_inputs = SQ_inputs.set_index('city')
SQ_inputs['PrEPCoverage'],  SQ_inputs['PrEPDuration']= 0, 0
input_bounds = h_fun1.load_json(r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/CEPAC RUNS/regression model input/input_mean_and_sd.json')

#%% STRATEGIES

# what strategies we want to simulate?
uptake = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
uptake_t = [24, 36, 48, 60]
uptake_s = [2]

# create dictionary of input parameters necessary to create an input signal
strategies = list(itertools.product(uptake, uptake_t, uptake_s))
FEATURE_VEC = h_fun1.get_feature_vector()
EXAMPLES, INPUT_DIM = len(strategies), len(FEATURE_VEC)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# initialize 
Run_A = h_fun2.build_state(SQ_inputs.loc['rio', :], input_bounds, DEVICE).repeat(EXAMPLES, 1, 1)
Run_B = torch.zeros((EXAMPLES, SRC_LEN, INPUT_DIM)).float().to(DEVICE)

# import all the cepac_outputs
trg_int, trg_int_std = h_fun2.create_target_tensor(TARGET_INT, DEVICE)
trg_sq, trg_sq_std = h_fun2.create_target_tensor(TARGET_SQ, DEVICE)
Y_int = torch.zeros((EXAMPLES, TRG_LEN, OUT_DIM)).float().to(DEVICE)
Y_sq = torch.zeros((EXAMPLES, TRG_LEN, OUT_DIM)).float().to(DEVICE)

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
        Run_B[idx_strategy, :, :] = h_fun2.build_state(INT_float.loc[city, :], input_bounds, DEVICE)
        
        # store targets
        Y_int[idx_strategy, :, :] = trg_int_std[(int(100 * cov), cov_t)]
        Y_sq[idx_strategy, :, :] = trg_sq_std[(0, 0)]


#%% COMMUNITY BEN

# initialize the model object
filepath = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Hyper parameter tuning/RNN_GRU.pt'#r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Hyper parameter tuning/VanillaRNN.pt'
z_path = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/CEPAC RUNS/regression model input'
Environment = FGRU(filepath)#FVan(filepath)

# dictionary for storage
ALL_per_red = {}
ALL_red_coe = {}

# forward pass
data = h_fun1.new_feature_representation([[Run_A, Y_sq]], data_type = 'torch')
prediction_sq = Environment(data, z_path, DEVICE)
data = h_fun1.new_feature_representation([[Run_B, Y_int]], data_type = 'torch')
prediction_int = Environment(data, z_path, DEVICE)

# iterate over all possible values (should be 24)
for strategy in range(0, len(strategies)):
    
    # TX algo
    percentage_reduction, reduction_coeff = h_fun2.community_benefit(prediction_sq['denormalized prediction'][0][:, strategy, :], prediction_int['denormalized prediction'][0][:, strategy, :])
    
    # save community benefit
    key = str(strategies[strategy])
    ALL_per_red[key], ALL_red_coe[key] = percentage_reduction, reduction_coeff
    
    
# plot heatmap from the saved values
    