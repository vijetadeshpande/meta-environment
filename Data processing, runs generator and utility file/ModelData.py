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
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data processing, runs generator and utility file')
import HelperFunctions1 as h_fun1
import os
import numpy as np
import torch
import pickle

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
        try:
            data_train = h_fun1.load_json(os.path.join(data_dir, 'train.json'))
            data_test = h_fun1.load_json(os.path.join(data_dir, 'test.json'))
            data_val = h_fun1.load_json(os.path.join(data_dir, 'validation.json'))
        except:
            try:
                infile = open(os.path.join(data_dir, 'train.pkl'), 'rb')
                data_train = pickle.load(infile)
                infile = open(os.path.join(data_dir, 'test.pkl'), 'rb')
                data_test = pickle.load(infile)
                infile = open(os.path.join(data_dir, 'validation.pkl'), 'rb')
                data_val = pickle.load(infile)
            except:
                try:
                    data_train = np.load(os.path.join(data_dir, 'train.npy'), allow_pickle = True).tolist()
                    data_test = np.load(os.path.join(data_dir, 'test.npy'), allow_pickle = True).tolist()
                    data_val = np.load(os.path.join(data_dir, 'validation.npy'), allow_pickle = True).tolist()
                except:
                    print('\nNO DATA FILE FOUND')
                    return
        
        # using new feature representation
        data_train = h_fun1.new_feature_representation(data_train)
        data_test = h_fun1.new_feature_representation(data_test)
        data_val = h_fun1.new_feature_representation(data_val)
        
        
        # STEP - 2: create numpy array of required shape and convert them into torch tensors
        X_train, Y_train = torch.FloatTensor(np.array(data_train[0][0])), torch.FloatTensor(np.array(data_train[0][1]))
        X_test, Y_test = torch.FloatTensor(np.array(data_test[0][0])), torch.FloatTensor(np.array(data_test[0][1]))
        X_val, Y_val = torch.FloatTensor(np.array(data_val[0][0])), torch.FloatTensor(np.array(data_val[0][1]))
        EXAMPLES_TRN, INPUT_SEQ_LEN, INPUT_DIM, EXAMPLES_TST, OUTPUT_SEQ_LEN = X_train.shape[0], X_train.shape[1], X_train.shape[-1], X_test.shape[0], Y_train.shape[1]
        
        # STEP - 3: create tuples
        train, test, val, train_run_idx, test_run_idx = [], [], [], [], []
        for example in range(0, EXAMPLES_TRN, batch_size):
            X, Y = X_train[example:example + batch_size, :, :], Y_train[example:example + batch_size, :]
            X, Y = X, Y#.unsqueeze(2)
            train.append((X, Y))
        for example in range(0, EXAMPLES_TST, batch_size):
            X, Y = X_test[example:example + batch_size, :, :], Y_test[example:example + batch_size, :]
            X, Y = X, Y#.unsqueeze(2)
            test.append((X, Y))
        for example in range(0, EXAMPLES_TST, batch_size):
            X, Y = X_val[example:example + batch_size, :, :], Y_val[example:example + batch_size, :]
            X, Y = X, Y#.unsqueeze(2)
            val.append((X, Y))
            
        # store output_dim
        OUTPUT_DIM = Y.shape[-1]
        
        # set attributes for the ModelData class
        self.train_examples, self.val_examples, self.test_examples = train, val, test
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
    
        
        
    

