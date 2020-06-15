#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 22:16:48 2020

@author: vijetadeshpande
"""

import torch
import sys
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data processing, runs generator and utility file')
import HelperFunctions1 as h_fun1
import pandas as pd
import os
from copy import deepcopy

def evaluate(model, data, criterion, device, seqpath):
    
    # initialize
    model.eval()
    epoch_loss = 0
    
    # collect all the outputs
    predictions = []
    denorm_predictions = []
    denorm_targets = []
    
    # import denormalization parameters
    #mean_sd = pd.read_csv(os.path.join(seqpath, 'output_mean_and_sd.csv'), header = 0, index_col = 0)
    
    with torch.no_grad():
        for example in data:
            
            # extract source and target
            source = data[0][0].to(device)
            target = data[0][1].to(device)
            
            # forward pass
            prediction = model(source, target)
            predictions.append(prediction.numpy())
            
            # denormalize prediction and append
            #denorm_prediction = h_fun1.denormalize(deepcopy(prediction.numpy()), mean_sd.iloc[0, :].values, mean_sd.iloc[1, :].values)
            #denorm_predictions.append(denorm_prediction)
            
            # denormalize target and save
            #trg = trg.permute(1, 0, 2)
            #denorm_target = h_fun1.denormalize(deepcopy(target.numpy()), mean_sd.iloc[0, :].values, mean_sd.iloc[1, :].values)
            #denorm_targets.append(denorm_target)
            
            # calculate error
            TRG_LEN, BATCH_SIZE, OUTPUT_DIM = prediction.shape
            prediction = torch.reshape(prediction[:, :, :], ((TRG_LEN)*BATCH_SIZE, OUTPUT_DIM))
            target = torch.reshape(target[:, 1:, :], ((TRG_LEN)*BATCH_SIZE, OUTPUT_DIM))
            
            # error calculation
            loss = criterion(target, prediction)
            
            # update error
            epoch_loss += loss.item()
    
    # return a dictionary
    all_metrics = {'average epoch loss': epoch_loss/len(data), 
                   'normalized prediction': predictions,
                   'denormalized prediction': denorm_predictions,
                   'denormalized target': denorm_targets}
            
    return all_metrics