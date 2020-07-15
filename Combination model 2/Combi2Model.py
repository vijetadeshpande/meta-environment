#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 23:57:13 2020

@author: vijetadeshpande
"""
import torch
import torch.nn as nn
from numpy import random


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout, device, is_bidirectional = False):
        super().__init__()
        
        # set attributes
        self.input_dim = input_dim
        n_directions = int(is_bidirectional) + 1
        self.hidden_dim = hidden_dim * n_directions
        self.n_layers = n_layers
        self.is_bidirectional = is_bidirectional
        self.device = device
        #self.teacher_forcing_ratio = teacher_forcing_ratio
        
        # dropout
        self.dropout = nn.Dropout(dropout)
        
        # rnn cell
        self.rnn = nn.RNN(input_size = input_dim,
                          hidden_size = hidden_dim,
                          num_layers = n_layers,
                          batch_first = True,
                          dropout = dropout)
        
        # dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # linear transformation to match required output dim
        self.linear_transform = nn.Linear(hidden_dim, output_dim)
        
        return
    
    def forward(self, source, target, teacher_forcing_ratio = 1):
        
        # input check
        # source = [batch size, seq len, input dim]
        # target = [batch size, seq len, output dim]
        
        # get shape 
        (BATCH_SIZE, SRC_LEN, INPUT_DIM) = source.shape
        (_, TRG_LEN, OUTPUT_DIM) = target.shape
        
        # rnn pass
        memory, _ = self.rnn(source)
        
        # dropout
        memory = self.dropout(memory)
        
        # inputs required for first time step
        prediction = self.linear_transform(memory)
    
        
        return prediction
    