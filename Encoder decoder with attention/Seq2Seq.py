#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 16:49:32 2020

@author: vijetadeshpande
"""
import torch
import torch.nn as nn
from numpy import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, attention = None, teacher_forcing_ratio = 0.5):
        super().__init__()
        
        # define encoder decoder attributes
        self.encoder = encoder
        self.attention = attention
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        #
        self.encoder_layers = encoder.n_layers
        self.decoder_layers = decoder.n_layers
        self.encoder_hidden = encoder.hidden_dim
        self.decoder_hidden = decoder.hidden_dim
        
        # linear transformation
        #self.junction = nn.Linear(encoder.hidden_dim, decoder.hidden_dim)
        
        #
        #assert encoder.hidden_dim == decoder.hidden_dim, \
        #    "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
            
    def forward(self, source, target):
        
        #src = [batch size, src len, input dim]
        #trg = [batch size, trg len, output dim]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # for the encoder (LSTM) we need the input of shape = (src len, batch size, input dim)
        source = source.permute(1, 0, 2).to(self.device)
        
        # for the decoder (LSTM) we need input of shape = (trg len, batch size, output dim)
        target = target.permute(1, 0, 2).to(self.device)
        
        # define tensor to store decoder outputs ()
        (SRC_LEN, BATCH_SIZE, INPUT_DIM) = source.shape
        (TRG_LEN, BATCH_SIZE, OUPTUT_DIM) = target.shape
        OUTPUT_DIM =  self.decoder.output_dim
        outputs = torch.zeros(TRG_LEN, BATCH_SIZE, OUTPUT_DIM, device = self.device)
        attention_ws = torch.zeros(BATCH_SIZE ,TRG_LEN, SRC_LEN, device = self.device)
        
        # compute cell state and hidden state from encoder forward pass
        if self.attention == None:
             _, hidden, cell = self.encoder(source)
        else:
            all_hidden, hidden, cell = self.encoder(source)
            
        # first input to decoder is sos tokens
        input = target[0, :] # therefore, in your processes data, first element in sequence should correspond to <sos>
        
        # linear transformation on hidden and cell state
        #hidden = self.junction(torch.reshape(hidden, (self.decoder_layers, BATCH_SIZE, self.encoder_hidden)))
        #cell = self.junction(torch.reshape(cell, (self.decoder_layers, BATCH_SIZE, self.encoder_hidden)))
        
        # an option to linear transformation is to only take first 'n_layers' only
        hidden = hidden[:self.encoder_layers, :, :]
        cell = cell[:self.encoder_layers, :, :]
        
        # loop over target sequence length
        for t in range(1, TRG_LEN):
                
            # forward pass of decoder (at first 't', hidden and cell will be 
            # taken from encoder output)
            if self.attention != None:
                output, hidden, cell, attention_w = self.decoder(input, hidden, cell, all_hidden)
                attention_ws[:, t, :] = attention_w[:, 0, :]
            else:
                output, hidden, cell, _ = self.decoder(input, hidden, cell)
            
            # store output
            outputs[t] = output
            
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < self.teacher_forcing_ratio
            
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = target[t] if teacher_force else output
        
        return outputs, attention_ws
        