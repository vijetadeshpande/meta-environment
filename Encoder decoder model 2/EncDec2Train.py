#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 13:55:06 2020

@author: vijetadeshpande
"""
import torch.nn
import torch

def train(model, data, optimizer, criterion, clip, device, cohort_size = 1):
    # initiliaze
    model.train()
    epoch_loss = 0
    BATCH_SIZE, SRC_LEN, _ = data[0][0].shape
    _, TRG_LEN, _ = data[0][1].shape
    attention_ws = torch.zeros((len(data), BATCH_SIZE, TRG_LEN-1, SRC_LEN)).to(device)
    
    # loop over all batches in iterator
    idx = -1
    for example in data:
        idx += 1
        
        # access the source and target sequence
        src = example[0].to(device)
        trg = example[1].to(device)
        
        # make gradients equal to zero
        optimizer.zero_grad()
        
        # forward pass through model
        prediction, attention_w = model(src, trg)
        CUR_SIZE, _, _ = attention_w.shape
        attention_ws[idx, 0:CUR_SIZE, :, :] = attention_w[:, 1:, :]
        
        # calculate error
        BATCH_SIZE, TRG_LEN, OUTPUT_DIM = prediction.shape
        prediction = torch.reshape(prediction[:, 1:, :], ((TRG_LEN - 1)*BATCH_SIZE, OUTPUT_DIM))
        trg = torch.reshape(trg[:, 1:, :], ((TRG_LEN - 1)*BATCH_SIZE, OUTPUT_DIM))
        
        # calculate loss
        loss = criterion(prediction, trg)
        
        # backward propogation
        loss.backward()
        
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # update weights
        optimizer.step()
        
        # update loss
        epoch_loss += loss.item()
    
    # take average of the loss
    epoch_loss = epoch_loss / len(data)
        
    
    return epoch_loss, attention_ws
        
    