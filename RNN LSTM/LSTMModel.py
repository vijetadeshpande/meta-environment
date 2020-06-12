#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 23:57:13 2020

@author: vijetadeshpande
"""
import torch
import torch.nn as nn
from numpy import random

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout, device, is_bidirectional = False, teacher_forcing_ratio = 0.5):
        super().__init__()
        
        # set attributes
        self.input_dim = input_dim
        n_directions = int(is_bidirectional) + 1
        self.hidden_dim = hidden_dim * n_directions
        self.n_layers = n_layers
        self.is_bidirectional = is_bidirectional
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        # dropout
        self.dropout = nn.Dropout(dropout)
        
        # rnn cell
        self.rnn = nn.LSTM(input_size = (input_dim + output_dim),
                          hidden_size = hidden_dim,
                          num_layers = n_layers,
                          batch_first = True,
                          dropout = dropout)
        
        # linear transformation to match required output dim
        stepdown1_dim, stepdown2_dim = max(1, int(hidden_dim/2)), max(1, int(hidden_dim/8))
        self.stepdown1 = nn.Linear(self.hidden_dim, stepdown1_dim)
        self.stepdown2 = nn.Linear(stepdown1_dim, stepdown2_dim)
        self.stepdown3 = nn.Linear(stepdown2_dim, output_dim)
        self.activation = nn.ReLU()
        
        return
    
    def forward(self, source, target):
        
        # input check
        # source = [batch size, seq len, input dim]
        # target = [batch size, seq len, output dim]
        
        # get shape 
        (BATCH_SIZE, SRC_LEN, INPUT_DIM) = source.shape
        (_, TRG_LEN, OUTPUT_DIM) = target.shape
    
        # tensor for storing outputs at each time step
        outputs = torch.zeros(TRG_LEN, BATCH_SIZE, OUTPUT_DIM).to(self.device)
        
        # inputs required for first time step
        hidden = torch.zeros(self.n_layers, BATCH_SIZE, self.hidden_dim)
        cell = torch.zeros(self.n_layers, BATCH_SIZE, self.hidden_dim)
        
        # iterate over the horizon
        for t in range(0, SRC_LEN):
            
            # use teacher forcing ratio
            if t == 0:
                teacher_force = True
            else:
                teacher_force = random.random() < self.teacher_forcing_ratio
            output = target[:, t, :].unsqueeze(1) if teacher_force else output
            
            # input for this time step (concatenate input and last output)
            input = source[:, t, :].unsqueeze(1)
            input = torch.cat((input, output), dim = 2)
            
            # forward pass through LSTM
            output, (hidden, cell) = self.rnn(input, (hidden, cell))
            
            # linear transformation of output
            output = self.activation(self.stepdown1(output))
            output = self.activation(self.stepdown2(output))
            output = self.stepdown3(output)
            
            # append
            outputs[t+1, :, :] = output.squeeze(1)

        
        return outputs
    