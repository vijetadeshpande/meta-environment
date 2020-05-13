#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 22:16:51 2020

@author: vijetadeshpande
"""

"""
Created on Wed Sep  4 16:53:34 2019

@author: Vijeta
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
import utils
import sys
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/CEPAC-extraction-tool')
import link_to_cepac_in_and_out_files as link
import TextFileOperations as t_op
import cluster_operations as c_op


"""
What are we changing in .in file for data generation

1. Incidence (~ Uniform(0.5/100PY, 5/100PY))
2. Prevalence (~ Uniform(0.5%, 30%))
    2.1 Sample #infected at index year (~ Uniform(1000, 20000))
    2.2 #susceptible = infected * (1-p)/p
3. Percetage on ART (~ Uniform(20%, 90%))
    3.1 Calculate community VL
    3.2 Calculate average transmission rate
    3.3 Calculate transmission rate multiplier
4. PrEP
    4.1 Sample coverage (~ Uniform(0, 100%))
    4.2 Sample coverage time (~ Uniform(0, 59))
    4.3 Fix the shape parameter = 2
    4.4 Weibull uptake probabilities

"""
#%% aux functions
def calculate_mean_sd(seq_out, seqpath):
    
    #
    mean, sd = {}, {}
    for feat in seq_out:
        mean[feat], sd[feat] = np.mean(seq_out[feat]), np.std(seq_out[feat])
    
    # save a csv file for mean and sd of the output
    par = pd.DataFrame(0, index = ['mean', 'sd'], columns = mean.keys())
    for var in mean:
        par.loc['mean', var] = mean[var]
        par.loc['sd', var] = sd[var]
    #par = par.drop(['multiplier'], axis = 1)
    par.to_csv(os.path.join(seqpath, 'output_mean_and_sd.csv'))

    return mean, sd

def zscore_standardization(seq_out, mean, sd):
    # from the calculated mean and standard deviation, this function
    # performs the Z-score normalization on the output data
    output_tensor = np.zeros((OUTPUT_FEATURES, EXAMPLES*N_FILES, SEQ_LEN))
    output_dict = {'transmissions': [], 'infections': [], 'susceptibles': []}
    for feat in seq_out:
        for example in range(0, len(seq_out[feat])):
            z_score = np.divide((seq_out[feat][example][:SEQ_LEN] - mean[feat]), sd[feat])
            output_dict[feat].append(z_score)
    for feat in output_dict:
        if feat == 'transmissions':
            output_tensor[0, :, :] = output_dict[feat]
        elif feat == 'infections':
            output_tensor[1, :, :] = output_dict[feat]
        elif feat == 'susceptibles':
            output_tensor[2, :, :] = output_dict[feat]
    
    return output_tensor#, run_labels

def take_cumulative_sum(list_in):
    EXAMPLES, SEQ_LEN, FEATURES = len(list_in), len(list_in[0]), len(list_in[0][0])
    
    for example in range(0, EXAMPLES):
        x = np.zeros((SEQ_LEN, FEATURES))
        x[:, :] = list_in[example]
        x = np.cumsum(x, axis = 1)
        list_in[example, :, :] = x
    
    return list_in

def padding(list_in):
    
    EXAMPLES, SEQ_LEN, FEATURES = len(list_in), len(list_in[0]), len(list_in[0][0])
    mat = np.zeros((EXAMPLES, SEQ_LEN + 2, FEATURES))
    
    for example in range(0, EXAMPLES):
        mat[:, 1:1+SEQ_LEN, :] = list_in[example]
    
    return mat.tolist()
        
        
#%% WRITING SEQUENCE GENERATOR RUNS
# collect samples
sample_n = 1000
samples, sample_bounds, var_list, parameters = ipar.get_samples(sample_n)
filepath = r'/Users/vijetadeshpande/Documents/GitHub/Sequence2Sequence model for CEPAC prediction/Data and results/basefile'
savepath = r'/Users/vijetadeshpande/Documents/GitHub/Sequence2Sequence model for CEPAC prediction/Data and results/CEPAC RUNS/NEW'
seqpath = r'/Users/vijetadeshpande/Documents/GitHub/Sequence2Sequence model for CEPAC prediction/sequence data as input'
INPUT_FEATURES = len(samples) - 4
OUTPUT_FEATURES = 3
EXAMPLES = sample_n
N_FILES = 1
SEQ_LEN = 60


if (not os.path.exists(os.path.join(savepath, 'results'))) and (not os.path.exists(os.path.join(savepath, str(0), 'results'))):
    # import base .in file
    basefile = link.import_all_cepac_in_files(filepath)['rio']
    
    # start timer
    start = timeit.default_timer()
    
    # loop over samples one by one to create .in files
    samples['Gaussian solution'] = []
    feature_tensor = []
    for run in range(0, sample_n):
        float_df = deepcopy(basefile)
        
        # loop over all the variables
        for var in var_list:
            # ignore few variables
            if var in ['onART', 'prevalence', 'PrEPEfficacy', 'PrEPAdherence', 'PrepIncidMale', 'InitAge mean', 'InitAge sd']:
                continue
            
            # else replace value
            elif var in ['HIVmthIncidMale']:
                # print
                #print('This variable has been altered in .in file: %s'%(var))
                
                # fit a Gaussian curve 
                #val, solution = utils.gaussian_fitter(samples[var][run], 
                #                                      p_init = [1, np.divide(samples['InitAge'][0][0], 12), np.divide(samples['InitAge'][0][1], 12)])
                val = samples[var][run]
                val = utils.expand_incidence(val, length = 8, step = 4)
                factor = 1 - np.multiply(parameters['PrEP efficacy'], parameters['PrEP adherence'])
                val_prep = np.multiply(factor, val)
                
                # store the solution
                #samples['Gaussian solution'].append(solution)
                
                # convert rate value to monthly probability
                val = utils.rate_to_prob(val, factor = 1200)
                val_prep = utils.rate_to_prob(val_prep, factor = 1200)
                
                # replace value in the .in file
                float_df = t_op.replace_values(var, val, float_df)
                float_df = t_op.replace_values('PrepIncidMale', val_prep, float_df)
                
                del val, val_prep#, solution
                
            else:
                # print
                #print('This variable has been altered in .in file: %s'%(var))
                
                # replace value in the .in file
                float_df = t_op.replace_values(var, samples[var][run], float_df)
        
        # save the file
        name = os.path.join(savepath, 'sample_' + str(run) + '.in')
        link.write_cepac_in_file(name, float_df)
        
        # featurizarion of the .in file
        feature_mat = utils.featurize_in_file(samples, sample_bounds, run, SEQ_LEN)
        feature_tensor.append(feature_mat)
        
        # print
        print(run)
        
    # parallelize input files
    c_op.parallelize_input(savepath, parallel = 5)
    
    # save feature tensor
    utils.dump_jsonl(feature_tensor, os.path.join(seqpath, 'input_tensor.json'))
    
    # stop timer
    stop = timeit.default_timer()
    
    print('Time: ', stop - start) 
    

    
    