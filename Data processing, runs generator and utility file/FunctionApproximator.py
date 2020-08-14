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
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Transformer')
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/RNN LSTM')
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/RNN Vanilla')
from GRUModel import Model as GRUModel
from GRUEvaluate import evaluate as GRUEval
from TransformerEncoder import Encoder as TEnc
from TransformerDecoder import Decoder as TDec
from TransformerModel import Model as TModel
from TransformerEvaluate import evaluate as TEval
from VanillaModel import Model as VanillaModel
from VanillaEvaluate import evaluate as VanillaEval
from LSTMModel import Model as LSTMModel
from LSTMEvaluate import evaluate as LSTMEval

class GRUApproximator(nn.Module):
    def __init__(self, filepath):
        super().__init__()
        
        # here we make a computational graph suited optimal RNN model
        # then we load the .pt file and we are all set
        
        # create model
        INPUT_DIM, OUTPUT_DIM = 8, 3
        HID_DIM, N_LAYERS, DROPOUT  = 24, 2, 0
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = GRUModel(INPUT_DIM, HID_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT, DEVICE)
        model = model.to(DEVICE)
        
        # load file
        model.load_state_dict(torch.load(filepath))
        model.eval()
        
        # set sttribute
        self.model = model
        
        #
        self.criterion = nn.MSELoss().to(DEVICE) #nn.SmoothL1Loss().to(DEVICE)
        
        return
    
    def forward(self, input_data, z_path, DEVICE):
        
        # check shape
        # input_tensor = [BATCH, SEQ_LEN, INPUT_DIM]
        #EXAMPLES, SRC_LEN, INPUT_DIM = input_tensor[0].shape
        
        # predict
        prediction = GRUEval(self.model, input_data, self.criterion, DEVICE, z_path)
        
        
        return prediction
        
class TransformerApproximator(nn.Module):
    def __init__(self, filepath):
        super().__init__()
        
        # create model
        INPUT_DIM, OUTPUT_DIM = 8, 3
        N_HEADS_ENC, N_HEADS_DEC, N_LAYERS, DROPOUT = 8, 3, 1, 0.05
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        # initialize encoder, decoder and seq2seq model classes
        encoder = TEnc(INPUT_DIM, OUTPUT_DIM, N_HEADS_ENC, N_LAYERS, DROPOUT)
        decoder = TDec(OUTPUT_DIM, N_HEADS_DEC, N_LAYERS, DROPOUT)
        model = TModel(encoder, decoder)
        model = model.to(DEVICE)
        
        # load parameters
        model.load_state_dict(torch.load(filepath))
        model.eval()
        
        # set attribute
        self.model = model
        self.criterion = nn.MSELoss().to(DEVICE)
        
        return
        
    def forward(self, input_data, z_path, DEVICE):
        
        # predicr
        prediction = TEval(self.model, input_data, self.criterion, DEVICE, z_path)
        
        return prediction
    
class VanillaApproximator(nn.Module):
    def __init__(self, filepath):
        super().__init__()
        
        # here we make a computational graph suited optimal RNN model
        # then we load the .pt file and we are all set
        
        # create model
        INPUT_DIM, OUTPUT_DIM = 8, 3
        HID_DIM, N_LAYERS, DROPOUT  = 32, 2, 0
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = VanillaModel(INPUT_DIM, HID_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT, DEVICE)
        model = model.to(DEVICE)
        
        # load file
        model.load_state_dict(torch.load(filepath))
        model.eval()
        
        # set sttribute
        self.model = model
        
        #
        self.criterion = nn.MSELoss().to(DEVICE) #nn.SmoothL1Loss().to(DEVICE)
        
        return
    
    def forward(self, input_data, z_path, DEVICE):
        
        # check shape
        # input_tensor = [BATCH, SEQ_LEN, INPUT_DIM]
        #EXAMPLES, SRC_LEN, INPUT_DIM = input_tensor[0].shape
        
        # predict
        prediction = VanillaEval(self.model, input_data, self.criterion, DEVICE, z_path)
        
        
        return prediction

class LSTMApproximator(nn.Module):
    def __init__(self, filepath):
        super().__init__()
        
        # here we make a computational graph suited optimal RNN model
        # then we load the .pt file and we are all set
        
        # create model
        INPUT_DIM, OUTPUT_DIM = 8, 3
        HID_DIM, N_LAYERS, DROPOUT  = 16, 2, 0
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = LSTMModel(INPUT_DIM, HID_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT, DEVICE)
        model = model.to(DEVICE)
        
        # load file
        model.load_state_dict(torch.load(filepath))
        model.eval()
        
        # set sttribute
        self.model = model
        
        #
        self.criterion = nn.MSELoss().to(DEVICE) #nn.SmoothL1Loss().to(DEVICE)
        
    def forward(self, input_data, z_path, DEVICE):
        
        # check shape
        # input_tensor = [BATCH, SEQ_LEN, INPUT_DIM]
        #EXAMPLES, SRC_LEN, INPUT_DIM = input_tensor[0].shape
        
        # predict
        prediction = LSTMEval(self.model, input_data, self.criterion, DEVICE, z_path)
        
        
        return prediction
        
        
        