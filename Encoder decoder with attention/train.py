#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 13:55:06 2020

@author: vijetadeshpande
"""
import torch.nn
import torch

def train(model, data, optimizer, criterion, clip, cohort_size = 1):
    # initiliaze
    model.train()
    epoch_loss = 0
    (BATCH_SIZE, SRC_LEN, _) = data[0][0].shape
    (_, TRG_LEN, _) = data[0][1].shape
    attention_ws = torch.zeros((len(data), BATCH_SIZE, TRG_LEN-1, SRC_LEN))
    
    # loop over all batches in iterator
    idx = -1
    for example in data:
        idx += 1
        
        # access the source and target sequence
        src = example[0]
        trg = example[1]
        
        # make gradients equal to zero
        optimizer.zero_grad()
        
        # feed src to the encoder to get cell and hidden
        # then feed cell and hidden to deoder to get the output
        output, attention_w = model(src, trg)
        CUR_SIZE, _, _ = attention_w.shape
        attention_ws[idx, 0:CUR_SIZE, :, :] = attention_w[:, 1:, :]
        
        #
        trg = trg.permute(1, 0, 2)
        # trg = [trg len, batch size, out dim]
        # output = [trg len, batch size, output dim]
        
        # calculate error
        TRG_LEN, BATCH_SIZE, OUTPUT_DIM = output.shape
        output = torch.reshape(output[1:, :, :], ((TRG_LEN - 1)*BATCH_SIZE, OUTPUT_DIM))
        trg = torch.reshape(trg[1:, :, :], ((TRG_LEN - 1)*BATCH_SIZE, OUTPUT_DIM))
        
        # calculate loss
        loss = criterion(output, trg)
        
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
        
    