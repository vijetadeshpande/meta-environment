#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 03:14:14 2020

@author: vijetadeshpande
"""
import os
import numpy as np
import itertools
import sys
import torch
from FunctionApproximator import GRUApproximator as FGRU
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data processing, runs generator and utility file') 
import utils2

#%% SOME PAR

DEVICE = 'cpu' 
SRC_LEN = 60
TRG_LEN = 61
SRC_DIM = 6
TRG_DIM = 3

#%% STRATEGIES

# what strategies we want to simulate?
uptake = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
uptake_t = [24, 36, 48, 60]
uptake_s = [2]
age = (30*12, 9.6*12)

# create dictionary of input parameters necessary to create an input signal
strategies = list(itertools.product(uptake, uptake_t, uptake_s))
input_par = {
    'InitAge': [], 
    'UseHIVIncidReduction': np.zeros(len(strategies)),
    'HIVIncidReductionStopTime': 60 * np.ones(len(strategies)),
    'HIVIncidReductionCoefficient': np.ones(len(strategies)),
    'HIVmthIncidMale': 4.3 * np.ones(len(strategies)),
    'PrEPEnable': np.ones(len(strategies)),
    'PrEPCoverage': [],
    'PrEPDuration': [],
    'PrEPShape': []
    }
for strategy in strategies:
    input_par['PrEPCoverage'].append(strategy[0])
    input_par['PrEPDuration'].append(strategy[1])
    input_par['PrEPShape'].append(strategy[2])
    input_par['InitAge'].append(age)

#%% COMMUNITY BEN

# initialize the model object
filepath = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/RNN GRU/best results/GRU_best.pt'
z_path = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/CEPAC RUNS/regression model input'
Environment = FGRU(filepath)

# dictionary for storage
ALL_per_red = {}
ALL_red_coe = {}

# iterate over all possible values (should be 24)
for strategy in range(0, len(strategies)):
    
    # get input tensor
    input_tensor = utils2.build_state(input_par, strategy)
    X = torch.tensor(input_tensor).type('torch.FloatTensor').to(DEVICE)
    Y = torch.zeros((1, TRG_LEN, TRG_DIM)).type('torch.FloatTensor').to(DEVICE)
    
    # SQ
    data = [(X, Y)]
    SQ_ = Environment(data, z_path, DEVICE)
    
    # INT
    X[:, :, 3] = 0
    data = [(X, Y)]
    INT_ = Environment(data, z_path, DEVICE)
    
    # TX algo
    percentage_reduction, reduction_coeff = utils2.community_benefit(SQ_['denormalized prediction'][0], INT_['denormalized prediction'][0])
    
    # save community benefit
    key = str(strategies[strategy])
    ALL_per_red[key], ALL_red_coe[key] = percentage_reduction, reduction_coeff
    
    
# plot heatmap from the saved values
    