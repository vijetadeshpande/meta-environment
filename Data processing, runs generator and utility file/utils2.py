#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 18:44:01 2020

@author: vijetadeshpande
"""

import scipy as sp
from copy import deepcopy
import os
import numpy as np
import utils
import _pickle as pickle
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import random

# build state of the system
def build_state(input_signal, example):
    
    # import distribution parameters (required for z-score calculations)
    #parameters = utils.load_json(r'input_bounds.json')
    
    # input signal should be a dictionary and must have following keys
    # 1. 'InitAge', 
    # 2. 'UseHIVIncidReduction', 
    # 3. 'HIVIncidReductionStopTime', 
    # 4. 'HIVIncidReductionCoefficient',
    # 5. 'HIVmthIncidMale',
    # 6. 'PrEPEnable'
    # 7. 'PrEPCoverage'
    # 8. 'PrEPDuration'
    # 9. 'PrEPShape'
    
    # the input signal to the RNN model right now is of shape [1, 60, 6] for one example
    EXAMPLES, SRC_LEN, INPUT_DIM = 1, 60, 6
    system_state = np.zeros((EXAMPLES, SRC_LEN, INPUT_DIM))
    
    # the system state should have following features in mentioned order
    # 1. Enable incidence reduction? this should always be zero while calculating community benefit
    # 2. Reduction coefficient. While calculating community benefit, ideally this should be infinity, just set it upperbound
    # 3. HIV incidence (monthly probability)
    # 4. Enable PrEP?
    # 5. Weibull uptake probabilities
    # 6. HIV incidence while on PrEP
    features = ['UseHIVIncidReduction', 'HIVIncidReductionCoefficient', 'HIVmthIncidMale', 'PrEPEnable', 'Weibull', 'PrepIncidMale']
    
    # iterate over the features
    first_visit = True
    feature_idx = -1
    for feature in features:
        feature_idx += 1
        if feature in ['HIVmthIncidMale', 'PrepIncidMale']:
            if first_visit:
                # update first visit
                first_visit = False
                
                # get incidence sequence
                inci = utils.get_incidence_sequence(input_signal, example, SRC_LEN)
                factor = 1 - np.multiply(0.739, 0.96)
                inci_prep = np.multiply(factor, inci)
                
                # set values
                system_state[:, :, feature_idx] = inci
                system_state[:, :, 5] = inci_prep
            else:
                continue
        elif feature == 'Weibull':
            system_state[:, :, feature_idx] = utils.weibull_tp(input_signal['PrEPCoverage'][example], input_signal['PrEPDuration'][example], input_signal['PrEPShape'][example], SRC_LEN)
        else:
            system_state[:, :, feature_idx] = input_signal[feature][example] 
                
    
    
    return system_state 

def community_benefit(run_A, run_B):
    
    # few fixed parameters
    EXAMPLES, TRG_LEN, OUT_DIM = run_A.shape
    COHORT_SIZE = 10000000
    
    
    # Utils
    def calculate_average_prob(total_inf):
        p = total_inf/COHORT_SIZE
        p_avg = 1 - np.power((1 - p), (1/TRG_LEN))
        
        return p_avg
    
    #
    idx = {'transmissions': 0,'infections': 1, 'susceptible': 2}
    
    
    # calculate difference bet infection cases in SQ and INV
    step_1 = np.sum(run_A[:, :, 0]) - np.sum(run_B[:, :, 0])
    # calculate average monthly incidence probability in SQ
    step_2 = calculate_average_prob(np.sum(run_A[:, :, 1]))
    # calculate average monthly incidence prob in INV
    step_3 = calculate_average_prob(np.sum(run_A[:, :, 1]) - step_1)
    # difference in avg monthly prob
    step_4 = step_3 - step_2
    # monthly prob at time 't' for INV
    inf_prob = np.divide(run_B[:, :, 1], run_B[:, :, 2])[TRG_LEN][0]
    step_5 = np.multiply(inf_prob, np.divide((np.sum(run_A[:, :, 1]) - step_1), np.sum(run_A[:, :, 1])))
    # percentage decrease in monthly probability at month t
    step_6 = np.divide((inf_prob - step_5), inf_prob)
    
    # incidence red
    percentage_decline = 100 * step_6
    # coefficient
    coeff = -1 * TRG_LEN/(np.log(1 - step_6))
    
    #
    output = {'Percentage reduction': percentage_decline, 'Reduction coefficient': coeff}
    
    # check
    #check = (((inf['RunA'] - tx['RunA']) < 0) or ((inf['RunB'] - tx['RunB']) < 0) or ((inf['RunC'] - tx['RunC']) < 0))
    #if check:
    #    print('wrong')
    """
    check = inf['RunA'] - tx['RunA']
    if check < 0:
        print('wrong')
    """
    
    return percentage_decline, coeff