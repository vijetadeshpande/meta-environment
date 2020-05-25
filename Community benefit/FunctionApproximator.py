#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 02:45:04 2020

@author: vijetadeshpande
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/RNN GRU')
from GRU import GRU
from evaluate import evaluate

class GRUApproximator(nn.Module):
    def __init__(self, filepath):
        super().__init__()
        
        # here we make a computational graph suited optimal RNN model
        # then we load the .pt file and we are all set
        
        # create model
        INPUT_DIM, OUTPUT_DIM = 6, 3
        HID_DIM, N_LAYERS, DROPOUT, DEVICE  = 512, 2, 0.5, 'cpu'
        model = GRU(INPUT_DIM, HID_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT, DEVICE)
        model = model.to(DEVICE)
        
        # load file
        model.load_state_dict(torch.load(filepath))
        
        # set sttribute
        self.model = model
        
        #
        self.criterion = nn.MSELoss().to(DEVICE) #nn.SmoothL1Loss()
        
        return
    
    def forward(self, input_data, z_path, DEVICE):
        
        # check shape
        # input_tensor = [BATCH, SEQ_LEN, INPUT_DIM]
        #EXAMPLES, SRC_LEN, INPUT_DIM = input_tensor[0].shape
        
        # initialize output
        prediction = evaluate(self.model, input_data, self.criterion, DEVICE, z_path)
        
        
        return prediction
        
        