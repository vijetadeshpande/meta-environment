#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:28:03 2020

@author: vijetadeshpande
"""

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from PositionalEncoding import PositionalEncoding

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads, n_layers, dropout):
        super().__init__()
        
        # positional encoding
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        
        # transformer encoder
        encoder_layer = TransformerEncoderLayer(input_dim*2, n_heads, dropout = dropout)
        self.encoder = TransformerEncoder(encoder_layer, n_layers)
        
        # linear transformation
        self.linear_transform = nn.Linear(input_dim*2, output_dim*2)
        
        # masking
        self.mask = None
        
        return
        
    def _generate_square_subsequent_mask(self, sz):
        
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        return mask
    
    def forward(self, source):
        
        # masking
        #if (self.mask is None) or (self.mask.size(0) != len(source)):
        #    mask = self._generate_square_subsequent_mask(len(source)).to(source.device)
        #    self.mask = mask
        
        # pos encoder pass
        source = self.pos_encoder(source)
        
        # forward pass
        encoder_out = self.encoder(source)#, self.mask)
        
        # linear transformation
        encoder_out = self.linear_transform(encoder_out)
        
        return encoder_out