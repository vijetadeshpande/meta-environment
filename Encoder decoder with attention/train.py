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
    
    # loop over all batches in iterator
    for example in data:
        
        # access the source and target sequence
        src = example[0]
        trg = example[1]
        
        # make gradients equal to zero
        optimizer.zero_grad()
        
        # feed src to the encoder to get cell and hidden
        # then feed cell and hidden to deoder to get the output
        output = model(src, trg)
        
        #
        trg = trg.permute(1, 0, 2)
        # trg = [trg len, batch size, out dim]
        # output = [trg len, batch size, output dim]
        
        # calculate error
        TRG_LEN, BATCH_SIZE, OUTPUT_DIM = output.shape
        output = torch.reshape(output[1:, :, :], ((TRG_LEN - 1)*BATCH_SIZE, OUTPUT_DIM))
        trg = torch.reshape(trg[1:, :, :], ((TRG_LEN - 1)*BATCH_SIZE, OUTPUT_DIM))
        loss = criterion(output, trg)
        
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
        
    
    return epoch_loss / len(data)
        
    