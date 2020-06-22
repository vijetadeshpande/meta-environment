#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:37:27 2020

@author: vijetadeshpande
"""

import torch
import torch.nn as nn
import math
 
class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, dropout = 0.1, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        
        # dropout
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, input_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_dim, 2).float() * (-math.log(10000.0) / input_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        if (input_dim % 2) != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        
        #
        pe = self.pe[:x.size(0), :, :].repeat(1, x.size(1), 1)
        x = torch.cat((x, pe), -1)
        #x += pe
        
        return self.dropout(x)
    