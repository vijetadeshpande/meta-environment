#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 23:57:13 2020

@author: vijetadeshpande
"""
import torch
import torch.nn as nn

class SimpleGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout, device, is_bidirectional = False):
        super().__init__()
        
        # set attributes
        self.input_dim = input_dim
        n_directions = int(is_bidirectional) + 1
        self.hidden_dim = hidden_dim * n_directions
        self.n_layers = n_layers
        self.is_bidirectional = is_bidirectional
        self.device = device
        
        # dropout
        self.dropout = nn.Dropout(dropout)
        
        # rnn cell
        self.rnn = nn.GRU(input_size = input_dim,
                          hidden_size = hidden_dim,
                          num_layers = n_layers,
                          batch_first = True,
                          dropout = dropout)
        
        # linear transformation to match required output dim
        self.transform = nn.Linear(self.hidden_dim, output_dim)
        
        return
    
    def forward(self, source, target):
        
        # input check
        # source = [batch size, seq len, input dim]
        # target = [batch size, seq len, output dim]
        
        # get shape 
        (BATCH_SIZE, SEQ_LEN, INPUT_DIM) = source.shape
        (_, _, OUTPUT_DIM) = target.shape
        
        # tensor for storing outputs at each time step
        outputs = torch.zeros(SEQ_LEN, BATCH_SIZE, OUTPUT_DIM).to(self.device)
        
        # inputs required for first time step
        hidden = torch.zeros(self.n_layers, BATCH_SIZE, self.hidden_dim)
        
        # iterate over the horizon
        for t in range(0, SEQ_LEN):
            
            # input for this time step
            input = source[:, t, :].unsqueeze(1)
            
            # forward pass through GRU
            output, hidden = self.rnn(input, hidden)
            
            # linear transformation of output
            output = self.transform(output)
            
            # append
            outputs[t, :, :] = output.squeeze(1)

        
        return outputs
    