#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:52:25 2020

@author: vijetadeshpande
"""
import scipy as sp
import numpy as np
from scipy.stats import truncnorm, bernoulli, uniform
import HelperFunctions1 as h_fun1
import math

#%% aux functions
def define_distribution(dict_in, distribution):
    if distribution == 'truncnorm':
        # define distribution parameters
        lower, upper, mu, sigma = dict_in['lb'], dict_in['ub'], dict_in['mean'], dict_in['sd']
        a = (lower - mu)/sigma
        b = (upper - mu)/sigma
        
        # define distribution object
        dist = truncnorm(a, b, loc = mu, scale = sigma)
        
    elif distribution == 'bernoulli':
        # this distribution is used for enabling following modules in CEPAC
        # 1. Incidence reduction due to community benefit
        # 2. PrEP
        # 3. Dynamic transmission module
        p = dict_in['p']
        
        # define distribution object
        dist = bernoulli(p)
    
    elif distribution == 'uniform':
        # define parameters
        lb, ub = dict_in['lb'], dict_in['ub']
        
        # define distribution
        dist = uniform(lb, ub)
    
    return dist

def calculate_average_tx_rate(on_ART, parameters):
    
    avg_rate = np.zeros((len(on_ART), ))
    idx = -1
    for val in on_ART:
        idx += 1
        community_VL = np.multiply(val, parameters['viral load distribution on ART']) + np.multiply((1 - val), parameters['viral load distribution off ART'])
        weighted_avg = np.sum(np.multiply(community_VL, parameters['attia']))
        avg_rate[idx] = weighted_avg
    
    return avg_rate
    
#%% MAIN FUNCTION
    
def get_samples(sample_n):
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
    sample_bounds = {}
    check = 0
    for var in var_list:
        if var in dep_var_list or var in ignore_var_list:
            if var == 'InitAge':
                sample_bounds[var] = {'lb': 196, 'ub': 480} # in months
            continue
        
        # age             
        elif var == 'InitAge mean':
            sample_bounds[var] = {'lb': 288,    #np.multiply(24, 12), 
                                 'ub': 420,     #np.multiply(35, 12),
                                 'mean': 360,   #np.multiply(30, 12), 
                                 'sd': 36}      #np.multiply(3, 12)
        elif var == 'InitAge sd':
             sample_bounds[var] = {'lb': 0,     #np.multiply(0, 12), 
                                  'ub': 120,    #np.multiply(10, 12),
                                  'mean': 60,   #np.multiply(5, 12), 
                                  'sd': 24}     #np.multiply(2, 12)
        
        # infection related parameters
        elif var == 'UseHIVIncidReduction':
            sample_bounds[var] = {'lb': 0, 'ub': 1,
                                 'p': 1}
        elif var ==  'HIVIncidReductionStopTime':
            sample_bounds[var] = {'lb': horizon, 'ub': horizon}
        elif var == 'HIVIncidReductionCoefficient':
            sample_bounds[var] = {'lb': 26, 'ub': 12000, # 12000 = 1%, 26 = 99%
                                 'mean': 5000, 'sd': 4900} 
        elif var == 'HIVmthIncidMale':
            sample_bounds[var] = {'lb': 1, 'ub': 5,
                                 'mean': 3, 'sd': 1.5}
        elif var == 'PrEPEnable':
            sample_bounds[var] = {'lb': 0, 'ub': 1,
                                 'p': 1}
        elif var == 'PrEPCoverage':
            sample_bounds[var] = {'lb': 0, 'ub': 0.80,
                                 'mean': 0.30, 'sd': 0.20}
        elif var == 'PrEPDuration':
            sample_bounds[var] = {'lb': 0, 'ub': 59, # in months
                                 'mean': 29, 'sd': 20}
        elif var == 'PrEPShape':
            sample_bounds[var] = {'lb': 2, 'ub': 2}
        
        # transmission related parameters
        elif var == 'DynamicTransmissionPropHRGAttrib':
            sample_bounds[var] = {'lb': 0.69, 'ub': 0.71,   #{'lb': 0.5, 'ub': 1,
                                  'mean': 0.7,'sd': 0}      #'mean': 0.7, 'sd': 0.2}
        elif var == 'prevalence':
            sample_bounds[var] = {'lb': 0.152, 'ub': 0.154, #{'lb': 0.001, 'ub': 0.20,
                                  'mean': 0.153,'sd': 0}    #'mean': 0.10, 'sd': 0.05}
        elif var == 'DynamicTransmissionNumHIVPosHRG':
            sample_bounds[var] = {'lb': 16998, 'ub': 17000, #{'lb': 1000, 'ub': 30000,
                                 'mean': 16999, 'sd': 0}    #'mean': 10000, 'sd': 8000}
        elif var == 'onART':
            sample_bounds[var] = {'lb': 0.72, 'ub': 0.74,   #{'lb': 30%, 'ub': 90%,
                                 'mean': 0.73, 'sd': 0}     #'mean': 50%, 'sd': 20%}
    
        else:
            continue
    
    #%% SAMPLING FOR INDEPENDENT VAR
    
    # define distribution and take samples
    samples = {}
    for var in sample_bounds:
        if var in fixed_var_list:
            samples[var] = sample_bounds[var]['mean'] * np.ones((sample_n))
        elif var in ['InitAge mean', 'InitAge sd', 'HIVIncidReductionCoefficient', 'HIVmthIncidMale', 'PrEPCoverage', 'PrEPDuration', 'onART', 'prevalence', 'DynamicTransmissionNumHIVPosHRG', 'DynamicTransmissionPropHRGAttrib']:
            sample_bounds[var]['distribution'] = define_distribution(sample_bounds[var], 'truncnorm')
            samples[var] = sample_bounds[var]['distribution'].rvs(sample_n)            
        elif var in ['UseHIVIncidReduction', 'PrEPEnable']:
            sample_bounds[var]['distribution'] = define_distribution(sample_bounds[var], 'bernoulli')
            samples[var] = sample_bounds[var]['distribution'].rvs(sample_n)
        elif var in ['HIVIncidReductionStopTime', 'PrEPShape', 'PrEPEfficacy', 'PrEPAdherence']:
            sample_bounds[var]['distribution'] = None
            samples[var] = sample_bounds[var]['lb'] * np.ones((sample_n, ))
    
    #%% CALCULATION OF SAMPLES FOR DEPENDENT VAR
    
    for var in dep_var_list:
        if var in ['TransmissionRiskMultiplier_T1', 'TransmissionRiskMultiplier_T2', 'TransmissionRiskMultiplier_T3']:
            continue
        elif var == 'InitAge':
            x = np.zeros((sample_n, 2))
            x[:, 0] = samples['InitAge mean']
            x[:, 1] = samples['InitAge sd']
            samples[var] = x
        elif var == 'PrepIncidMale':
            x = samples['HIVmthIncidMale']
            factor = 1 - np.multiply(parameters['PrEP efficacy'], parameters['PrEP adherence'])
            x = np.multiply(factor, x)
            samples[var] = x
        elif var == 'DynamicTransmissionNumHIVNegHRG':
            factor = np.divide((1 - samples['prevalence']), samples['prevalence'])
            x = np.multiply(samples['DynamicTransmissionNumHIVPosHRG'], factor)
            samples[var] = x
            # save parameters
            #mean_X, mean_Y, var_X, var_Y = sample_bounds['prevalence']['mean'], sample_bounds['DynamicTransmissionNumHIVPosHRG']['mean'], np.power(sample_bounds['prevalence']['sd'], 2), np.power(sample_bounds['DynamicTransmissionNumHIVPosHRG']['sd'], 2)
            #mean_Z = mean_X * mean_Y
            #var_Z = (np.power(mean_X, 2) + var_X) * (np.power(mean_Y, 2) + var_Y) - (np.power(mean_X, 2) * np.power(mean_Y, 2))
            #sample_bounds[var] = {'mean': mean_Z, 'sd': np.power(var_Z, 0.5)}
        elif var == 'DynamicTransmissionNumTransmissionsHRG':
            x = calculate_average_tx_rate(samples['onART'], parameters)
            samples[var] = x
            # save parameters
            sample_bounds[var] = {'mean': np.mean(x), 'sd': np.std(x)}
            
    
    #%% FEW ADJUSTMENTS
    
    # 1. PrEP duration, PrEP shape should be integers
    # 2. PrEP coverage should be divided by 100
    for var in var_list:
        if var in ['PrEPDuration', 'PrEPShape']:
            samples[var] = samples[var].astype(int)
    
    # final var list
    final_var_list = [i for i in samples]
    
    # following change is temporary
    # as we are keeping HRG_POS and HRG_NEG constant then the z-stand will not
    # work as sd = 0. Hence, we use probable values of mean and sd here
    sample_bounds['DynamicTransmissionNumHIVPosHRG']['mean'], sample_bounds['DynamicTransmissionNumHIVPosHRG']['sd'] = 10000, 8000
    sample_bounds['DynamicTransmissionNumTransmissionsHRG']['mean'], sample_bounds['DynamicTransmissionNumTransmissionsHRG']['sd'] = 4.1350, 0.8472
            
    return samples, sample_bounds, final_var_list, parameters


# try running this
#_, bounds, _, _ = get_samples(1000)