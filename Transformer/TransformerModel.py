#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:16:27 2020

@author: vijetadeshpande
"""

import torch
import torch.nn as nn
import random

class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
        return
    
    def forward(self, source, target, module = 'train'):
        
        # dim
        TRG_LEN, BATCH_SIZE, TRG_DIM = target.shape
        
        # encoder pass
        memory = self.encoder(source)
        
        # decoder pass
        prediction = self.decoder(target, memory, module)
            
        
        return prediction
        
        