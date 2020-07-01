#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 00:15:26 2020

@author: vijetadeshpande
"""
import torch.nn as nn
import torch.nn.functional as F
import torch

class Attention(nn.Module):
    def __init__(self, encoder, dec_hidden_dim, n_layers):
        super().__init__()
        
        # attention transfomation layer
        enc_hidden_dim = encoder.hidden_dim
        nn_input_dim = enc_hidden_dim + dec_hidden_dim
        nn_output_dim = dec_hidden_dim
        self.attn = nn.Linear(nn_input_dim, nn_output_dim)
        
        # learnable par
        self.v = nn.Linear(dec_hidden_dim, 1, bias = False)
        
        return
        
    def forward(self, dec_hidden, enc_outputs):
        
        # input dim check
        # dec_hidden = [num_layers, batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        # dim
        BATCH_SIZE = dec_hidden.shape[1]
        SRC_LEN = enc_outputs.shape[0]
        
        # repeat decoder hidden state for src_len times 
        dec_hidden = dec_hidden[-1, :, :].repeat(SRC_LEN, 1, 1).permute(1, 0, 2)#.unsqueeze(0)
        enc_outputs = enc_outputs.permute(1, 0, 2)
        
        # dec_hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        # linear transformation
        energy = torch.tanh(self.attn(torch.cat((dec_hidden, enc_outputs), dim = 2)))
        #energy = [batch size, src len, dec hid dim]
        
        # attention
        attention = self.v(energy).squeeze(2)        
        # pass through softmax to get probability distribution over the 
        # sequence len for each example in batch
        attention = F.softmax(attention, dim=1)
        
        # output dim check
        # attention= [batch size, src len]
        
        return attention
    
class CosineAttention(nn.Module):
    def __init__(self, encoder, dec_hidden_dim, n_layers):
        super().__init__()
        
        # dimentional properties
        self.enc_hidden_dim = encoder.hidden_dim
        self.enc_is_bidirectional = encoder.is_bidirectional
        
        # in this case hidden dim of encoder and decoder should be same
        self.similarity = nn.CosineSimilarity()
        
        
        return
    
    def forward(self, dec_hidden, enc_outputs):
        
        # input dim check
        # dec_hidden = [num_layers, batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        SRC_LEN = enc_outputs.shape[0]
        if self.enc_is_bidirectional:
            enc_hidden = self.enc_hidden_dim/2
        else:
            enc_hidden = self.enc_hidden_dim
        
        # reshape
        enc_outputs = enc_outputs[:, :, 0:enc_hidden].permute(1, 2, 0)
        dec_hidden = dec_hidden[-1, :, :].repeat(SRC_LEN, 1, 1).permute(1, 2, 0)
        
        # calculate similarity
        sim = self.similarity(enc_outputs, dec_hidden)
        attention = F.softmax(sim, dim = 1)
        
        return attention
        