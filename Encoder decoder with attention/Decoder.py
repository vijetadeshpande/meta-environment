#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 11:41:37 2020

@author: vijetadeshpande
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, n_layers, dropout, encoder, attention = None):
        super().__init__()
        
        #setting different dimension attributes
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.attention = attention
        
        # define type of unit cell
        enc_hidden_dim = encoder.hidden_dim * int(attention != None)
        self.rnn = nn.LSTM(input_size = (enc_hidden_dim + output_dim), 
                           hidden_size = hidden_dim, 
                           num_layers = n_layers, 
                           dropout = dropout)
        
        # define dropout
        self.dropout = nn.Dropout(dropout)
        
        # define tranformation on output
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        return
        
    def forward(self, input, hidden, cell, enc_outputs = None):
        
        # opposed to encoder, this forward pass will be initiated with two
        # inputs cell state and hidden state
        
        # input dim check:
        # input = [batch size, output_dim]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        
        # unsqueeze input to match the shape [1, batch_size, output_dim]
        input = input.unsqueeze(0)
        
        # initiate alpha
        alphas = None
        
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
            input = torch.cat((context, input), dim = 2)
            
        # dropout
        input = self.dropout(input)
        
        # LSTM input dim check
        # input = [seq len, batch size, input feature size]
        # hidden and cell = [layers * direction, batch, hidden size]
        
        # forward pass with rnn
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        
        # transformation on output
        prediction = self.fc_out(output.squeeze(0))
        
        # output dim check
        # prediction = [batch size, output feature size]
        
        return prediction, hidden, cell, alphas
        
        
        