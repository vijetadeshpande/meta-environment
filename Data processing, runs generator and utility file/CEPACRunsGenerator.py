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
        
#%% WRITING SEQUENCE GENERATOR RUNS

# few parameters
EXAMPLES = 1000
SEQ_LEN = 60

# collect samples
samples, sample_bounds, var_list, parameters = ipar.get_samples(EXAMPLES)
filepath = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/basefile'
savepath = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/CEPAC RUNS/NEW BATCH'

# create folder to save the files
savepath_cepac = os.path.join(savepath, 'Files for CEPAC')
savepath_rnn = os.path.join(savepath, 'Files for RNN')
for i in [savepath_cepac, savepath_rnn]:
    if not os.path.exists(i):
        os.makedirs(i)


if (not os.path.exists(os.path.join(savepath, 'results'))) and (not os.path.exists(os.path.join(savepath, str(0), 'results'))):
    # import base .in file
    basefile = link.import_all_cepac_in_files(filepath)['rio']
    
    # start timer
    start = timeit.default_timer()
    
    # loop over samples one by one to create .in files
    #samples['Gaussian solution'] = []
    feature_tensor = []
    CEPAC_input_condensed = pd.DataFrame(-10, index = np.arange(EXAMPLES), columns = var_list)
    for run in range(0, EXAMPLES):
        float_df = deepcopy(basefile)
        # loop over all the variables
        for var in var_list:
            
            # store input value
            if var != 'InitAge':
                CEPAC_input_condensed.loc[run, var] = samples[var][run]
            
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
        name = os.path.join(savepath_cepac, 'sample_' + str(run) + '.in')
        link.write_cepac_in_file(name, float_df)
        
        # featurizarion of the .in file
        feature_mat = utils.featurize_in_file(samples, sample_bounds, run, SEQ_LEN)
        feature_tensor.append(feature_mat)
        
        # print
        print(run)
        
    # save condensed cepac input files
    CEPAC_input_condensed = CEPAC_input_condensed.to_dict()
    utils.dump_json(CEPAC_input_condensed, os.path.join(savepath_rnn, 'CEPAC_input.json'))
    
    # parallelize input files
    c_op.parallelize_input(savepath_cepac, parallel = 5)
    
    # save feature tensor
    if not os.path.exists(savepath_rnn):
        os.makedirs(savepath_rnn)
    utils.dump_json(feature_tensor, os.path.join(savepath_rnn, 'RNN_source.json'))
    
    # stop timer
    stop = timeit.default_timer()
    
    print('Time: ', stop - start) 
    

    
    