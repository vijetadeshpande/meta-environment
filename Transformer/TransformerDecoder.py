#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:12:22 2020

@author: vijetadeshpande
"""

import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from PositionalEncoding import PositionalEncoding
import random

class Decoder(nn.Module):
    def __init__(self, output_dim, n_heads, n_layers, dropout):
        super().__init__()
        
        # positional encoding
        self.pos_encoder = PositionalEncoding(output_dim, dropout)
        
        # transformer encoder
        decoder_layer = TransformerDecoderLayer(output_dim*2, n_heads, dropout = dropout)
        self.decoder = TransformerDecoder(decoder_layer, n_layers)
        
        # activation after decoder
        self.linear_transform = nn.Linear(output_dim*2, output_dim)
        
        # masking
        self.mask = None
        
        return
        
    def _generate_square_subsequent_mask(self, sz):
        
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == False, float('-inf')).masked_fill(mask == True, float(0.0))
        
        return mask
    
    def forward(self, target, memory, module = 'train', teacher_forcing_ratio = 0):
        
        # masking
        if (self.mask is None) or (self.mask.size(0) != len(target)):
            mask = self._generate_square_subsequent_mask(len(target)).to(target.device)
            self.mask = mask
        
        # pos encoder pass
        TRG_LEN, BATCH_SIZE, TRG_DIM = target.shape
        target = self.pos_encoder(target)
        
        if module == 'train':
        
            # forward pass
            decoder_out = self.decoder(target, memory, tgt_mask = self.mask)
            
            # end actication
            decoder_out = self.linear_transform(decoder_out)
            
        else:
            
            # slice out pe
            pe = target[:, :, TRG_DIM:]
            
            # initialize
            prediction_prev = target[0, :, :].unsqueeze(0)
            decoder_out = torch.zeros(TRG_LEN, BATCH_SIZE, TRG_DIM).float().to(target.device)
            
            # iterate over target sequence length
            for t in range(0, TRG_LEN-1):
                
                # use teacher forcing ratio
                if t == 0:
                    teacher_force = True
                else:
                    teacher_force = random.random() < teacher_forcing_ratio
                prediction_prev = target[t, :, :].unsqueeze(0) if teacher_force else prediction_prev
                
                
                # decoder pass for current t in sequence len
                prediction_cur = self.decoder(prediction_prev, memory, tgt_mask = None)
                prediction_cur = self.linear_transform(prediction_cur)
                
                # store value
                decoder_out[t+1, :, :] = prediction_cur
                
                # update prediction prev
                prediction_prev = torch.cat((prediction_cur, pe[t+1, :, :].unsqueeze(0)), -1)
                
                
        
        return decoder_out