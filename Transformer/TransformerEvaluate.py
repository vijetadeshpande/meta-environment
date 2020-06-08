#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 10:02:15 2020

@author: vijetadeshpande
"""

import torch
import sys
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data processing, runs generator and utility file')
import utils
import pandas as pd
import os
from copy import deepcopy


def evaluate(model, data, criterion, device, seqpath):
    
    # initialize
    model.eval()
    epoch_loss = 0

    
    # import denormalization parameters
    mean_sd = pd.read_csv(os.path.join(seqpath, 'output_mean_and_sd.csv'), header = 0, index_col = 0)
    
    # initialize tensor to store attention weights
    N_BATCHES = len(data)
    BATCH_SIZE, SRC_LEN, _ = data[0][0].shape
    _, TRG_LEN, OUT_DIM = data[0][1].shape
    attention_ws = torch.zeros((N_BATCHES, BATCH_SIZE, TRG_LEN-1, SRC_LEN)).float().to(device)
    
    # collect all the outputs
    outputs = torch.zeros((N_BATCHES, TRG_LEN-1, BATCH_SIZE, OUT_DIM)).float().to(device)
    denorm_outputs = torch.zeros((N_BATCHES, TRG_LEN-1, BATCH_SIZE, OUT_DIM)).float().to(device)
    denorm_targets = torch.zeros((N_BATCHES, TRG_LEN-1, BATCH_SIZE, OUT_DIM)).float().to(device)
    
    
    with torch.no_grad():
        idx = -1
        for example in data:
            idx += 1
            
            # access the source and target sequence
            src = example[0].permute(1, 0, 2).to(device)
            trg = example[1].permute(1, 0, 2).to(device)
            
            # predict output and append
            output = model(src, trg)#, 0) # switch off teacher forcing
            
            # store results
            _, CUR_LEN, _ = output.shape
            outputs[idx, :, 0:CUR_LEN , :] = output[1:, :, :]
            #CUR_SIZE, _, _ = attention_w.shape
            #attention_ws[idx, 0:CUR_SIZE, :, :] = attention_w[:, 1:, :]
            
            # denormalize prediction and store
            denorm_output = utils.denormalize(deepcopy(output), mean_sd.iloc[0, :].values, mean_sd.iloc[1, :].values)
            denorm_outputs[idx, :, 0:CUR_LEN, :] = denorm_output[1:, :, :]
            
            # denormalize target and save
            denorm_target = utils.denormalize(deepcopy(trg), mean_sd.iloc[0, :].values, mean_sd.iloc[1, :].values)
            denorm_targets[idx, :, 0:CUR_LEN, :] = denorm_target[1:, :, :]
            
            # dimension check:
            # trg = [target_len, batch_size, out dim]
            # output = [target_len, batch_size, out dim]
            
            # calculate error
            output = torch.reshape(output[1:, :, :], ((TRG_LEN - 1)*CUR_LEN, OUT_DIM))
            trg = torch.reshape(trg[1:, :, :], ((TRG_LEN - 1)*CUR_LEN, OUT_DIM))
            loss = criterion(output, trg)
            
            # update error
            epoch_loss += loss.item()
    
    # return a dictionary
    all_metrics = {'average epoch loss': epoch_loss/len(data), 
                   'normalized prediction': outputs,#.permute(0, 2, 1, 3),
                   'denormalized prediction': denorm_outputs,#.permute(0, 2, 1, 3),
                   'denormalized target': denorm_targets,#.permute(0, 2, 1, 3),
                   'attention weights': attention_ws}
            
    return all_metrics