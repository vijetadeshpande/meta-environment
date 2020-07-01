#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 11:41:37 2020

@author: vijetadeshpande
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, output_dim, dec_hidden_dim, n_layers, dropout, encoder, attention = None):
        super().__init__()
        
        #setting different dimension attributes
        self.hidden_dim = dec_hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.attention = attention
        
        # define type of unit cell
        #enc_hidden_dim = encoder.hidden_dim * int(attention != None)
        self.rnn = nn.RNN(input_size = encoder.hidden_dim, 
                           hidden_size = dec_hidden_dim, 
                           num_layers = n_layers, 
                           dropout = dropout,
                           batch_first = True)
        
        # define dropout
        self.dropout = nn.Dropout(dropout)
        
        # linear transformation to match required output dim
        self.ff_stepdown = nn.Linear(dec_hidden_dim, output_dim)
        stepdown1_dim, stepdown2_dim = max(1, int(dec_hidden_dim/2)), max(1, int(dec_hidden_dim/8))
        self.stepdown1 = nn.Linear(self.hidden_dim, stepdown1_dim)
        self.stepdown2 = nn.Linear(stepdown1_dim, stepdown2_dim)
        self.stepdown3 = nn.Linear(stepdown2_dim, output_dim)
        #self.activation = nn.ReLU()
        
        
        return
        
    def forward(self, target, hidden, enc_outputs = None):
        
        # initiate alpha
        alphas = None
        
        # if we are using attention
        if self.attention != None:
            # calculate attention over input sequence
            alphas = self.attention(hidden, enc_outputs)
            # change the shape to [batch_size, 1, src_len]
            alphas = alphas.unsqueeze(1)
            # change the shape of the enc_outputs, from [src_len, batch_size, hid_dim] to [batch_size, src_len, hid_dim]
            enc_outputs = enc_outputs.permute(1, 0, 2)
            # use attention to weight encoder outputs
            context = torch.bmm(alphas, enc_outputs).permute(1, 0, 2)
            # concatenate the input and the context 
            target = torch.cat((context, target), dim = 2)
            
        # dropout
        target = self.dropout(target)
        
        # forward pass with rnn
        prediction, hidden = self.rnn(target, hidden)
        
        # linear transformation on output
        #prediction = self.ff_stepdown(output)
        
        # output dim check
        # prediction = [batch size, output feature size]
        
        return prediction, hidden, alphas
        
        
        