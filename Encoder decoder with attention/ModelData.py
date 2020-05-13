#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:13:50 2020

@author: vijetadeshpande
"""
import pandas as pd
from torchtext import data
import torch
import sys
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/Sequence2Sequence model for CEPAC prediction/Data processing, runs generator and utility file')
import utils
import os
import numpy as np
import torch

"""Encapsulates DataLoaders and Datasets for training, validation, test. 
Base class for fastai *Data classes."""

# create tuple of source and target
class BatchTuple():
    def __init__(self, dataset, source, target):
        self.dataset, self.source, self.target = dataset, source, target
        
    def __iter__(self):
        for batch in self.dataset:
            source = getattr(batch, self.source) 
            target = getattr(batch, self.target)                 
            yield (source, target)
            
    def __len__(self):
        return len(self.dataset)

class ModelData():
    
    def __init__(self, data_dir, batch_size):
        
        # step by step data preprocessing
        # 1. Read the training and testing data
        # 2. Convert data into required shape and convert data into FloatTensor
        # 3. Create tuples of source and target, i.e. (source, target)
        # 4. Create batches of examples
        
        # STEP - 1: read data
        data_train = utils.load_json(os.path.join(data_dir, 'train.json'))
        data_test = utils.load_json(os.path.join(data_dir, 'test.json'))
        
        # STEP - 2: create numpy array of required shape and convert them into torch tensors
        X_train, Y_train = torch.FloatTensor(np.array(data_train[0][0])), torch.FloatTensor(np.array(data_train[0][1]))
        X_test, Y_test = torch.FloatTensor(np.array(data_test[0][0])), torch.FloatTensor(np.array(data_test[0][1]))
        EXAMPLES_TRN, INPUT_SEQ_LEN, INPUT_DIM, EXAMPLES_TST, OUTPUT_SEQ_LEN = X_train.shape[0], X_train.shape[1], X_train.shape[-1], X_test.shape[0], Y_train.shape[1]
        
        # STEP - 3: create tuples
        train, test, train_run_idx, test_run_idx = [], [], [], []
        for example in range(0, EXAMPLES_TRN, batch_size):
            X, Y = X_train[example:example + batch_size, :, :], Y_train[example:example + batch_size, :]
            X, Y = X, Y#.unsqueeze(2)
            train.append((X, Y))
        for example in range(0, EXAMPLES_TST, batch_size):
            X, Y = X_test[example:example + batch_size, :, :], Y_test[example:example + batch_size, :]
            X, Y = X, Y#.unsqueeze(2)
            test.append((X, Y))
            
        # store output_dim
        OUTPUT_DIM = Y.shape[-1]
        
        # set attributes for the ModelData class
        self.train_examples, self.val_examples, self.test_examples = train, None, test
        #self.SRC, self.TRG = SRC, TRG
        
        # dimension attributes
        self.input_features, self.output_features = INPUT_DIM, OUTPUT_DIM
        self.batch_size = batch_size
    
        return

    @classmethod
    def from_dls(cls, path,trn_dl,val_dl,test_dl=None):
        #trn_dl,val_dl = DataLoader(trn_dl),DataLoader(val_dl)
        #if test_dl: test_dl = DataLoader(test_dl)
        return cls(path, trn_dl, val_dl, test_dl)

    @property
    def is_reg(self): return self.trn_ds.is_reg
    @property
    def is_multi(self): return self.trn_ds.is_multi
    @property
    def trn_ds(self): return self.trn_dl.dataset
    @property
    def val_ds(self): return self.val_dl.dataset
    @property
    def test_ds(self): return self.test_dl.dataset
    @property
    def trn_y(self): return self.trn_ds.y
    @property
    def val_y(self): return self.val_ds.y
    

