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
import pandas as pd
import torch
import HelperFunctions1 as h_fun1
import _pickle as pickle
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import sys
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/CEPAC-extraction-tool')
import link_to_cepac_in_and_out_files as link

# build state of the system
def build_state(input_signal, sample_bounds, DEVICE = 'cpu', output_type = 'torch'):
    
    # import distribution parameters (required for z-score calculations)
    #parameters = h_fun1.load_json(r'input_bounds.json')
    
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
    FEATURE_VEC = h_fun1.get_feature_vector()
    EXAMPLES, SRC_LEN, INPUT_DIM = 1, 60, len(FEATURE_VEC)
    system_state = pd.DataFrame(0, index = np.arange(SRC_LEN), columns = FEATURE_VEC)
    
    # the system state should have following features in mentioned order
    # 1. Enable incidence reduction? this should always be zero while calculating community benefit
    # 2. Reduction coefficient. While calculating community benefit, ideally this should be infinity, just set it upperbound
    # 3. HIV incidence (monthly probability)
    # 4. Enable PrEP?
    # 5. Weibull uptake probabilities
    # 6. HIV incidence while on PrEP
    
    # iterate over the features
    for var in FEATURE_VEC:
        
        # switch like features (or the features which will have same value for SEQ_LEN)
        if var in ['UseHIVIncidReduction', 'PrEPEnable', 'HIVIncidReductionCoefficient', 'DynamicTransmissionNumTransmissionsHRG', 'DynamicTransmissionPropHRGAttrib', 'DynamicTransmissionNumHIVPosHRG', 'prevalence']:
            
            # following if loop is specially for old .in files
            # where we did not save the prevalence value in the input dictionary
            if var == 'prevalence' and input_signal[var] == -10:
                input_signal[var] = input_signal['DynamicTransmissionNumHIVPosHRG']/(input_signal['DynamicTransmissionNumHIVPosHRG'] + input_signal['DynamicTransmissionNumHIVNegHRG'])
            
            val = h_fun1.expand_input(input_signal[var], SRC_LEN)
            if (var == 'HIVIncidReductionCoefficient'):
                if (input_signal['HIVIncidReductionStopTime'] < SRC_LEN-1):
                    # after reduction_stop_time, the factor changes to max value
                    val[input_signal['HIVIncidReductionStopTime']:] = sample_bounds['HIVIncidReductionStopTime']['ub']
                # calculate z-score of the val
                val = h_fun1.z_score(val, sample_bounds['HIVIncidReductionCoefficient']['mean'], sample_bounds['HIVIncidReductionCoefficient']['sd'])
            elif var in ['DynamicTransmissionNumTransmissionsHRG', 'DynamicTransmissionNumHIVPosHRG']:
                # calculate z-score
                val = h_fun1.z_score(val, sample_bounds[var]['mean'], sample_bounds[var]['sd'])
        
            
        # weibull uptake probabilities
        elif var == 'PrEPCoverage':
            target_uptake, target_time, shape = input_signal['PrEPCoverage'], input_signal['PrEPDuration'], input_signal['PrEPShape']
            val = h_fun1.weibull_tp(target_uptake, target_time, shape, SRC_LEN)
        
        # generating sequence of the incidence 
        elif var == 'HIVmthIncidMale':
            # calculate sequence of raw incidence
            val = h_fun1.get_incidence_sequence(input_signal, 0, SRC_LEN)
            
            # TODO: following line is hard coded (need to take reference for eff and adhe)
            factor = 1 - np.multiply(0.96, 0.739)
            
            # calculate sequence of prep incidence
            val_prep = np.multiply(factor, val)
            system_state.loc[:, 'PrepIncidMale'] = val_prep
            
        elif var in ['PrepIncidMale', 'DynamicTransmissionNumHIVNegHRG']:
            continue
        
        # store
        system_state.loc[:, var] = val
    
    # convert to torch tensor
    if output_type == 'torch':
        system_state = torch.tensor(system_state.values).type('torch.FloatTensor').to(DEVICE)
    else:
        system_state = system_state.values.tolist()
    
    
    return system_state 

def community_benefit(run_A, run_B):
    
    #
    if isinstance(run_A, torch.Tensor):
        run_A = deepcopy(run_A).numpy()
        run_B = deepcopy(run_B).numpy()
    
    # few fixed parameters
    TRG_LEN, OUT_DIM = run_A.shape
    COHORT_SIZE = 10000000
    
    
    # Utils
    def calculate_average_prob(total_inf):
        p = total_inf/COHORT_SIZE
        p_avg = 1 - np.power((1 - p), (1/TRG_LEN))
        
        return p_avg
    
    #
    idx = {'transmissions': 0,'infections': 1, 'susceptible': 2}
    
    
    # calculate difference bet infection cases in SQ and INV
    step_1 = np.sum(run_A[:, 0]) - np.sum(run_B[:, 0])
    # calculate average monthly incidence probability in SQ
    step_2 = calculate_average_prob(np.sum(run_A[:, 1]))
    # calculate average monthly incidence prob in INV
    step_3 = calculate_average_prob(np.sum(run_B[:, 1]) - step_1)
    # difference in avg monthly prob
    step_4 = step_3 - step_2
    # monthly prob at time 't' for INV
    inf_prob = np.divide(run_B[:, 1], run_B[:, 2])[TRG_LEN-1]
    step_5 = np.multiply(inf_prob, np.divide((np.sum(run_A[:, 1]) - step_1), np.sum(run_A[:, 1])))
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

def create_target_tensor(path, device, SEQ_LEN = 60):
    
    # path: result folder of the set of runs for which we want to create the target tensor
    
    # import all results
    cepac_outputs = link.import_all_cepac_out_files(path, module = 'regression')
    targets, targets_std = {}, {}
    
    # import output distribtion parameters
    para = pd.read_csv(r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/CEPAC RUNS/regression model input/output_mean_and_sd.csv').set_index('Unnamed: 0')
    
    # iterate over each file
    for file in cepac_outputs:
        # skip run_c
        if 'RunC' in file:
            continue
        
        # get coverage and coverage time, for setting it as key of dict
        try:
            coverage, cov_time = int(file.split('=')[1][:2]), int(file.split('=')[2])
        except:
            if 'SQ' in file: 
                coverage, cov_time = 0, 0
            else:
                coverage, cov_time = 'NA', 'NA'
        
        # remove the multiplier array
        try:
            cepac_outputs[file].pop('multiplier')
        except:
            pass
        
        #
        for feature in cepac_outputs[file]:
            cepac_outputs[file][feature] = cepac_outputs[file][feature].to_list()[:SEQ_LEN]
        
        # store as matrix
        targets[(coverage, cov_time)] = pd.DataFrame(0, index = np.arange(SEQ_LEN + 1), columns = cepac_outputs[file].keys())
        targets[(coverage, cov_time)].iloc[1:, :] = pd.DataFrame(cepac_outputs[file]).values
        
        # standardize targets
        targets_std[(coverage, cov_time)] = pd.DataFrame(0, index = targets[(coverage, cov_time)].index, columns = targets[(coverage, cov_time)].columns)
        for feature in cepac_outputs[file]:
            targets_std[(coverage, cov_time)].loc[1:, feature] = (targets[(coverage, cov_time)].loc[1:, feature] - para.loc['mean', feature])/para.loc['sd', feature]
            
        # convert to numpy array 
        targets[(coverage, cov_time)] = torch.tensor(np.array(targets[(coverage, cov_time)])).float().to(device)
        targets_std[(coverage, cov_time)] = torch.tensor(np.array(targets_std[(coverage, cov_time)])).float().to(device)
    
    return targets, targets_std
    
    