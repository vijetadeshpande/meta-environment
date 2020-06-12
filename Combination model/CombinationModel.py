#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 00:54:58 2020

@author: vijetadeshpande
"""

import torch
import torch.nn as nn
from numpy import random
import sys
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Encoder decoder with attention')
from Encoder import Encoder as LSTMEnc
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Transformer')
from TransformerEncoder import Encoder as TEnc
from TransformerDecoder import Decoder as TDec


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers_base, dropout_base, n_heads, n_layers_transformer, dropout_transformer):
        super().__init__()
        
        # create a base LSTM layer to transform the source and traget to a 
        # different space via the hidden state of the LSTM layer.
        self.base_source = LSTMEnc(input_dim, hidden_dim, n_layers_base, dropout_base)
        self.base_target = LSTMEnc(output_dim, hidden_dim, n_layers_base, dropout_base)
        
        # the transformed source and target are then fed to the tranformer layer
        self.transformer_enc = TEnc(hidden_dim, hidden_dim, n_heads, n_layers_transformer, dropout_transformer)
        self.transformer_dec = TDec(hidden_dim, n_heads, n_layers_transformer, dropout_transformer)
        
        # final linear transformation to required size of the output
        stepdown1_dim, stepdown2_dim = max(1, int(hidden_dim/2)), max(1, int(hidden_dim/8))
        self.stepdown1 = nn.Linear(hidden_dim, stepdown1_dim)
        self.stepdown2 = nn.Linear(stepdown1_dim, stepdown2_dim)
        self.stepdown3 = nn.Linear(stepdown2_dim, output_dim)
        self.activation = nn.ReLU()
        
        return
    
    def forward(self, source, target):
        
        # shape = [N, T, D]
        
        # first we'll pass both source and target through base LSTM layer
        source, _, _ = self.base_source(source)
        target, _, _ = self.base_target(target)
        
        # change the shape of the output for transformer pass
        source = source.permute(1, 0, 2)
        target = target.permute(1, 0, 2)
        
        # now we'll use expanded source and target for transformer pass
        memory = self.transformer_enc(source)
        prediction = self.transformer_dec(source, memory)
        
        # now get the prediction in required shape
        prediction = self.activation(self.stepdown1(prediction))
        prediction = self.activation(self.stepdown2(prediction))
        prediction = self.stepdown3(prediction)
        
        return prediction