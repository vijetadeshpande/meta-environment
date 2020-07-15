#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 00:49:12 2020

@author: vijetadeshpande
"""

from GRUModel import Model
from GRUTrain import train
from GRUEvaluate import evaluate
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import seaborn as sns
import os
import math
import sys
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data processing, runs generator and utility file')
import HelperFunctions1 as h_fun1
from ModelData import ModelData


# path variables
#datapath = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/CEPAC RUNS/regression model input'
#respath = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/RNN results'

#HID_DIM = 512
#N_LAYERS = 3
#DROPOUT = 0.5
#l_rate = 0.001
#N_EPOCHS = 10
#DEVICE = 'cpu'

def init_training(data_object, par_dict, datapath, respath):
    
    # unroll parameter values
    HID_DIM = par_dict['hidden dimension']
    N_LAYERS = par_dict['number of layers']
    DROPOUT = par_dict['dropout rate']
    L_RATE = par_dict['learning rate']
    N_EPOCHS = par_dict['number of epochs']
    DEVICE = par_dict['device']
    
    # create data object
    #data_object = ModelData(datapath, batch_size = 128)
    data_train, data_test, data_val = data_object.train_examples, data_object.test_examples, data_object.val_examples
    
    # parameters for defining encoder and decoder
    INPUT_DIM, OUTPUT_DIM = data_object.input_features, data_object.output_features
    
    # initialize encoder, decoder and seq2seq model classes
    model = Model(INPUT_DIM, HID_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT, DEVICE)
    model = model.to(DEVICE)
    
    # initialize values of learnable parameters
    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)
    model.apply(init_weights)
    
    # count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nThe model has {count_parameters(model):,} trainable parameters')
    
    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr = L_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.3)
    
    # define error function (ignore padding and sos/eos tokens)
    #TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    criterion = nn.MSELoss() #nn.CosineSimilarity(dim = 2) #nn.SmoothL1Loss() #nn.MSELoss()
    criterion = criterion.to(DEVICE)
    
    # training parameters
    CLIP = 1
    best_valid_loss = float('inf')
    
    # auxilliary function
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    
    # start training without attention
    train_losses = []
    valid_losses = []
    total_time = 0
    for epoch in range(N_EPOCHS):
        
        # WITHOUT ATTENTION
        # start clock
        start_time = time.time()
        # train
        train_loss = train(model, data_train, optimizer, criterion, CLIP, DEVICE)
        valid_loss = evaluate(model, data_val, criterion, DEVICE, datapath)['average epoch loss']
        # stop clock
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # print
        # if we happen to have a validation data set then calculate validation
        # loss here by predicting the value of validation set 'x's

        # update validation loss if less than previously observed minimum
        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'RNN_GRU.pt')        
    
        # print progress
        try:
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.4f} | Train PPL: {math.exp(train_loss):7.4f}')
            print(f'\t Val. Loss: {valid_loss:.4f} |  Val. PPL: {math.exp(valid_loss):7.4f}')
        except:
            print('Error is too large')
            
        # store error value
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        # update time
        total_time += (epoch_mins + (epoch_secs/60))
        
        # update learning rate
        scheduler.step()
    
    
    # shuffle the dataset and calculate error on training set again
    data_train_s = h_fun1.tensor_shuffler(data_train, DEVICE)
    start_time = time.time()
    prediction_train_s = evaluate(model, data_train_s, criterion, DEVICE, datapath)
    tt_loss_s = prediction_train_s['average epoch loss']
    end_time = time.time()
    pred_mins, pred_secs = epoch_time(start_time, end_time)
    
    
    # testing/prediction
    #model.load_state_dict(torch.load('tut1-model.pt'))
    prediction = evaluate(model, data_test, criterion, DEVICE, datapath)
    test_loss = prediction['average epoch loss']
    
    # save predicted values
    #filename = os.path.join(respath, 'GRU_RNN_test_result_samples.json')
    #utils.dump_json([prediction['denormalized prediction'][0].tolist(), prediction['denormalized target'][0].tolist()], filename)
    
    print(f'| Test Loss: {test_loss:.4f} | Test PPL: {math.exp(test_loss):7.4f} |')
    
    # save df for lineplot
    plot_df = pd.DataFrame(train_losses, columns = ['Mean Squared Error'])
    plot_df['Epoch'] = np.arange(len(plot_df))
    #plot_df.to_csv(os.path.join(respath, 'RNN GRU_Error plot.csv'))
    
    
    return {'train loss': train_losses, 
            'validation loss': valid_losses,
            'shuffle train loss': tt_loss_s, 
            'test loss': test_loss, 
            'total time': total_time}
