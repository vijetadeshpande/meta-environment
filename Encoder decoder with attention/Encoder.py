#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 21:49:35 2020

@author: vijetadeshpande
"""

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, is_bidirectional = False):
        super().__init__()
        
        # set attributes for the encoder
        self.n_layers = n_layers
        self.is_bidirectional = is_bidirectional
        
        # NOTE: if encoder is bidirectional, then the hidden state will be of
        # dimension = hidden_dim * 2. If want to take reference of this dimension
        # at some other place, then it needs to be altered as follows.
        # BUT
        # we pass hidden_dim (and not self.hidden_dim) for defining our unit 
        # rnn cell
        factor = int(is_bidirectional) + 1
        self.hidden_dim = hidden_dim * factor
        
        # TODO: what if we want to use pre-trained vectors
        # define embeddings
        #self.embedding = nn.Embedding(input_dim, embedding_dim)
        
        # define unit cell of encoder, this is usually GRU or LSTM
        self.rnn = nn.LSTM(input_size = input_dim, 
                           hidden_size = hidden_dim, 
                           num_layers = n_layers, 
                           dropout = dropout,
                           bidirectional = is_bidirectional)
        
        # define dropout
        self.dropout = nn.Dropout(dropout)
        
        return
        
    def forward(self, source):
        
        # input dim check
        # source = [source_len, batch_size, features]
        
        # dropout
        source = self.dropout(source)
        
        # take this embedding representation of the current example and pass it
        # into the LSTM
        output, (hidden, cell) = self.rnn(source)
        
        # output dim check
        # out = [source_len, batch_size, hid_dim*n_directions]
        # hidden = [n_layers*n_directions, batch_size, hid_dim]
        # cell = [n_layers*n_directions, batch_size, hid_dim]
        
        return output, hidden, cell
