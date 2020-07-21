#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 14:47:27 2020

@author: vijetadeshpande
"""
import numpy as np
import HelperFunctions1 as h_fun1


def get_featurized_input(action, community_benefit = [[0, 0, 0], [60, 60, 60], [12000, 12000, 12000]]):
    # some required parameters
    horizon = 60
    parameters = {'attia': np.array([9.03, 8.12, 8.12, 4.17, 2.06, 0.16, 0, 62.56]),
                  'viral load distribution off ART': np.array([0.4219, 0.1956, 0.1563, 0.1201, 0.0825, 0.0235, 0.0000, 0.0000]),
                  'viral load distribution on ART': np.array([0.0003, 0.0036, 0.0217, 0.0963, 0.3787, 0.4994, 0.0000, 0.0000]),
                  'PrEP efficacy': 0.96,
                  'PrEP adherence': 0.74,
                  'rio': {'viral load distribution': np.array([0.1141, 0.0554, 0.0580, 0.1027, 0.2987, 0.3709, 0.0000, 0.0000]),
                          'incidence': np.array([4.3, 1]),
                          'index positive': 16999,
                          'index negative': 94104,
                          'on ART': 0.73
                          },
                  'salvador': {'viral load distribution': np.array([0.1226, 0.0593, 0.0607, 0.1032, 0.2928, 0.3614, 0.0000, 0.0000]),
                          'incidence': np.array([2.45, 1]),
                          'index positive': 3926,
                          'index negative': 41728,
                          'on ART': 0.71
                          },
                  'manaus': {'viral load distribution': np.array([0.1268, 0.0612	, 0.0621, 0.1034, 0.2898, 0.3566, 0.0000, 0.0000]),
                          'incidence': np.array([1.4, 1]),
                          'index positive': 2828,
                          'index negative': 45937,
                          'on ART': 0.70
                          }
                  }
    
    #%% LIST OF VARIABLES 
    
    # these are the variables we need to change in order to generate the train
    # and test data. Some of the varables are independent of other variables but
    # some are dependent. Sampling is not done for dependent variable, they're
    # simply calculated from the sampled values of independent variables
    var_list = ['InitAge', 'InitAge mean', 'InitAge sd',
                'UseHIVIncidReduction', 'HIVIncidReductionStopTime', 'HIVIncidReductionCoefficient',
                'HIVmthIncidMale',
                'PrEPEnable', 'PrEPCoverage', 'PrEPDuration', 'PrEPShape', 'PrEPEfficacy', 'PrEPAdherence',
                'PrepIncidMale',
                'onART', 'prevalence',
                'DynamicTransmissionPropHRGAttrib', 'DynamicTransmissionNumHIVPosHRG', 'DynamicTransmissionNumHIVNegHRG',
                'DynamicTransmissionNumTransmissionsHRG', 'TransmissionRiskMultiplier_T1', 'TransmissionRiskMultiplier_T2', 'TransmissionRiskMultiplier_T3',
                ]
    cont_var_list = ['HIVIncidReductionStopTime', 
                     'PrEPShape', 'PrEPEfficacy', 'PrEPAdherence']
    dep_var_list = ['InitAge',
                    'PrepIncidMale', 
                    'DynamicTransmissionNumHIVNegHRG',
                    'DynamicTransmissionNumTransmissionsHRG', 'TransmissionRiskMultiplier_T1', 'TransmissionRiskMultiplier_T2', 'TransmissionRiskMultiplier_T3']
    # let's ignore the transmission related variables in the simple version
    ignore_var_list = ['TransmissionRiskMultiplier_T1', 'TransmissionRiskMultiplier_T2', 'TransmissionRiskMultiplier_T3']
    fixed_var_list = ['DynamicTransmissionPropHRGAttrib', 'prevalence', 'DynamicTransmissionNumHIVPosHRG', 'onART']
    
    #%% STORE PARAMETERS FOR DEFINING PROB. DIST.
    
    # store parameters for each variable, which then will be used to define 
    # probability distribution for respective variable
    samples = {}
    check = 0
    for var in var_list:
        if var in dep_var_list or var in ignore_var_list:
            continue
        
        # age             
        elif var == 'InitAge mean':
            samples[var] = [360, 348, 300]
        
        elif var == 'InitAge sd':
             samples[var] = [115.2, 97.2, 70.8]
        
        # infection related parameters
        elif var == 'UseHIVIncidReduction':
            samples[var] = community_benefit[0]
            
        elif var ==  'HIVIncidReductionStopTime':
            samples[var] = community_benefit[1]
            
        elif var == 'HIVIncidReductionCoefficient':
            samples[var] = community_benefit[2]
            
        elif var == 'HIVmthIncidMale':
            samples[var] = [4.3, 2.45, 1.4]
            
        elif var == 'PrEPEnable':
            samples[var] = action['PrEP enable']
            
        elif var == 'PrEPCoverage':
            samples[var] = action['PrEP coverage']
            
        elif var == 'PrEPDuration':
            samples[var] = action['PrEP duration']
            
        elif var == 'PrEPShape':
            samples[var] = action['PrEP shape']
        
        # transmission related parameters
        elif var == 'DynamicTransmissionPropHRGAttrib':
            samples[var] = [0.7, 0.7, 0.7]
            
        elif var == 'prevalence':
            samples[var] = [0.153, 0.153, 0.153]
            
        elif var == 'DynamicTransmissionNumHIVPosHRG':
            samples[var] = [16999, 16999, 16999]
            
        elif var == 'onART':
            samples[var] = [0.73, 0.73, 0.73]
    
        else:
            continue
    
    
    #%%
    for var in dep_var_list:
        if var in ['TransmissionRiskMultiplier_T1', 'TransmissionRiskMultiplier_T2', 'TransmissionRiskMultiplier_T3']:
            continue
        elif var == 'InitAge':
            x = np.zeros((len(samples['InitAge mean']), 2))
            x[:, 0] = samples['InitAge mean']
            x[:, 1] = samples['InitAge sd']
            samples[var] = x
        elif var == 'PrepIncidMale':
            x = samples['HIVmthIncidMale']
            factor = 1 - np.multiply(parameters['PrEP efficacy'], parameters['PrEP adherence'])
            x = np.multiply(factor, x)
            samples[var] = x
        elif var == 'DynamicTransmissionNumHIVNegHRG':
            factor = np.divide((1 - np.array(samples['prevalence'])), samples['prevalence'])
            x = np.multiply(samples['DynamicTransmissionNumHIVPosHRG'], factor)
            samples[var] = x
            # save parameters
            #mean_X, mean_Y, var_X, var_Y = sample_bounds['prevalence']['mean'], sample_bounds['DynamicTransmissionNumHIVPosHRG']['mean'], np.power(sample_bounds['prevalence']['sd'], 2), np.power(sample_bounds['DynamicTransmissionNumHIVPosHRG']['sd'], 2)
            #mean_Z = mean_X * mean_Y
            #var_Z = (np.power(mean_X, 2) + var_X) * (np.power(mean_Y, 2) + var_Y) - (np.power(mean_X, 2) * np.power(mean_Y, 2))
            #sample_bounds[var] = {'mean': mean_Z, 'sd': np.power(var_Z, 0.5)}
        elif var == 'DynamicTransmissionNumTransmissionsHRG':
            x = h_fun1.calculate_average_tx_rate(samples['onART'], parameters)
            samples[var] = x

    # import sample bounds
    sample_bounds = h_fun1.load_json(r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/CEPAC RUNS/regression model input/input_mean_and_sd.json')
    
    # featurization
    feature_tensor = []
    for i in range(len(samples['InitAge mean'])):
        
        # featurizarion of the .in file
        feature_mat = h_fun1.featurize_in_file(samples, sample_bounds, i, horizon)
        feature_tensor.append(feature_mat)
    
    #
    feature_tensor = h_fun1.new_feature_representation([[feature_tensor, []]])[0][0]
    
    return feature_tensor


# test call
action_ = {'PrEP enable': [0, 0, 0], 'PrEP coverage': [0.3, 0.3, 0.3], 
          'PrEP duration': [36, 36, 36], 'PrEP shape': [2, 2, 2]}
f_input1 = np.array(get_featurized_input(action_))
    
action_ = {'PrEP enable': [1, 1, 1], 'PrEP coverage': [0.3, 0.3, 0.3], 
          'PrEP duration': [36, 36, 36], 'PrEP shape': [2, 2, 2]}
f_input2 = np.array(get_featurized_input(action_))

# compare
compare = f_input1 - f_input2
