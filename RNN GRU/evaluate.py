#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 20:47:35 2020

@author: vijetadeshpande
"""
import torch
import sys
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/Sequence2Sequence model for CEPAC prediction/Data processing, runs generator and utility file')
import utils
import pandas as pd
import os
from copy import deepcopy

def evaluate(model, data, criterion, seqpath):
    
    # initialize
    model.eval()
    epoch_loss = 0
    
    # collect all the outputs
    outputs = []
    denorm_outputs = []
    denorm_targets = []
    
    # import denormalization parameters
    mean_sd = pd.read_csv(os.path.join(seqpath, 'output_mean_and_sd.csv'), header = 0, index_col = 0)
    
    with torch.no_grad():
        for example in data:
            
            # access the source and target sequence
            src = example[0]
            trg = example[1]
            
            # predict output and append
            output = model(src, trg)#, 0) # switch off teacher forcing
            outputs.append(output.numpy())
            
            # denormalize prediction and append
            denorm_output = utils.denormalize(deepcopy(output.numpy()), mean_sd.iloc[0, :].values, mean_sd.iloc[1, :].values)
            denorm_outputs.append(denorm_output)
            
            # denormalize target and save
            trg = trg.permute(1, 0, 2)
            denorm_target = utils.denormalize(deepcopy(trg.numpy()), mean_sd.iloc[0, :].values, mean_sd.iloc[1, :].values)
            denorm_targets.append(denorm_target)
            
            # dimension check:
            # trg = [target_len, batch_size, out dim]
            # output = [target_len, batch_size, out dim]
            
            # calculate error
            TRG_LEN, BATCH_SIZE, OUTPUT_DIM = output.shape
            output = torch.reshape(output[:, :, :], ((TRG_LEN - 0)*BATCH_SIZE, OUTPUT_DIM))
            trg = torch.reshape(trg[1:, :, :], ((TRG_LEN - 0)*BATCH_SIZE, OUTPUT_DIM))
            loss = criterion(output, trg)
            
            # update error
            epoch_loss += loss.item()
    
    # return a dictionary
    all_metrics = {'average epoch loss': epoch_loss/len(data), 
                   'normalized prediction': outputs,
                   'denormalized prediction': denorm_outputs,
                   'denormalized target': denorm_targets}
            
    return all_metrics
            