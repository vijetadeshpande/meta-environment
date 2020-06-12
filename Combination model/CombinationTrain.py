#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 01:37:52 2020

@author: vijetadeshpande
"""

import torch.nn
import torch

def train(model, data, optimizer, criterion, clip, device):
    
    # initialize
    model.train()
    epoch_loss = 0
    
    for example in data:
        
        # extract source and target
        source = data[0][0].to(device)
        target = data[0][1].to(device)
        
        # reset gradient
        optimizer.zero_grad()
        
        # forward pass
        prediction = model(source, target)
        
        # reshape for error calculation
        TRG_LEN, BATCH_SIZE, OUTPUT_DIM = prediction.shape
        prediction = torch.reshape(prediction[:, :, :], ((TRG_LEN)*BATCH_SIZE, OUTPUT_DIM))
        target = torch.reshape(target[:, 1:, :], ((TRG_LEN)*BATCH_SIZE, OUTPUT_DIM))
        
        # error calculation
        loss = criterion(target, prediction)
        
        # backprop
        loss.backward()
        
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # update weights
        optimizer.step()
        
        # update loss
        epoch_loss += loss.item()
        
    # take average of the loss
    epoch_loss = epoch_loss / len(data)
    
    return epoch_loss