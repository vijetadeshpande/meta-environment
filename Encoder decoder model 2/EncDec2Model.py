#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 16:49:32 2020

@author: vijetadeshpande
"""
import torch
import torch.nn as nn
from numpy import random

class Model(nn.Module):
    def __init__(self, encoder, decoder, device, attention = None):
        super().__init__()
        
        # define encoder decoder attributes
        self.encoder = encoder
        self.attention = attention
        self.decoder = decoder
        self.device = device
        #self.teacher_forcing_ratio = teacher_forcing_ratio
        
        #
        self.encoder_layers = encoder.n_layers
        self.decoder_layers = decoder.n_layers
        self.encoder_hidden = encoder.hidden_dim
        self.decoder_hidden = decoder.hidden_dim
        
        # linear transformation
        self.ff_stepdown = nn.Linear(decoder.hidden_dim, decoder.output_dim)
        self.ff_stepup = nn.Linear(decoder.output_dim, decoder.hidden_dim)
        
        #
        #assert encoder.hidden_dim == decoder.hidden_dim, \
        #    "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
            
    def forward(self, source, targets, teacher_forcing_ratio = 1):
        
        # define tensor to store decoder outputs ()
        (BATCH_SIZE, SRC_LEN, INPUT_DIM) = source.shape
        (BATCH_SIZE, TRG_LEN, OUPTUT_DIM) = targets.shape
        OUTPUT_DIM = self.decoder.output_dim
        
        # expand target space
        targets = self.ff_stepup(targets)
        
        # storage tensors
        predictions = torch.zeros(BATCH_SIZE, TRG_LEN, OUTPUT_DIM).to(self.device)
        attention_ws = torch.zeros(BATCH_SIZE ,TRG_LEN, SRC_LEN).to(self.device)
        
        # compute cell state and hidden state from encoder forward pass
        if self.attention == None:
             _, memory_h = self.encoder(source)
        else:
            all_hidden_enc, memory_h = self.encoder(source)
            
        # first input to decoder is sos tokens
        target = targets[:, 0, :].unsqueeze(1) # therefore, in your processes data, first element in sequence should correspond to <sos>
        #target = memory_h.permute(1, 0, 2)
        
        # initiate hidden and cell state
        hidden = memory_h#torch.zeros(memory_h.shape).float().to(self.device)
        
        
        # loop over target sequence length
        for t in range(1, TRG_LEN):
                
            # forward pass of decoder (at first 't', hidden and cell will be 
            # taken from encoder output)
            if self.attention != None:
                prediction, hidden, cell, attention_w = self.decoder(target, hidden, all_hidden_enc)
                attention_ws[:, t, :] = attention_w[:, 0, :]
            else:
                prediction, hidden, _ = self.decoder(target, hidden)
            
            # store output
            predictions[:, t, :] = self.ff_stepdown(prediction)[:, 0, :]
            
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            target = targets[:, t, :].unsqueeze(1) if teacher_force else prediction
        
        return predictions, attention_ws
        