#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:30:18 2020

@author: vijetadeshpande
"""
from TransformerModel import Model
from TransformerEncoder import Encoder
from TransformerDecoder import Decoder
from TransformerEvaluate import evaluate
from train import train
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
import utils
from ModelData import ModelData


# path variables
datapath = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/CEPAC RUNS/regression model input'
respath = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/RNN results'

# create data object
data_object = ModelData(datapath, batch_size = 64)
data_train, data_test = data_object.train_examples, data_object.test_examples

# parameters for defining encoder and decoder
INPUT_DIM, OUTPUT_DIM = data_object.input_features, data_object.output_features
N_HEADS_ENC, N_HEADS_DEC = 4, 3
N_LAYERS = 2
DROPOUT = 0.4
DEVICE = 'cpu'

# initialize encoder, decoder and seq2seq model classes
encoder = Encoder(INPUT_DIM, OUTPUT_DIM, N_HEADS_ENC, N_LAYERS, DROPOUT)
decoder = Decoder(OUTPUT_DIM, N_HEADS_DEC, N_LAYERS, DROPOUT)
model = Model(encoder, decoder)
model = model.to(DEVICE)

# initialize values of learnable parameters
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
model.apply(init_weights)

# count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} trainable parameters')

# define optimizer
optimizer = optim.Adam(model.parameters())

# define error function (ignore padding and sos/eos tokens)
#TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.MSELoss() #nn.CosineSimilarity(dim = 2) #nn.SmoothL1Loss() #nn.MSELoss()
criterion = criterion.to(DEVICE)

# training parameters
N_EPOCHS = 20
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
for epoch in range(N_EPOCHS):
    
    # WITHOUT ATTENTION
    # start clock
    start_time = time.time()
    # train
    train_loss = train(model, data_train, optimizer, criterion, CLIP, DEVICE)
    # stop clock
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    # print
    # if we happen to have a validation data set then calculate validation
    # loss here by predicting the value of validation set 'x's
    valid_loss = 0
    # update validation loss if less than previously observed minimum
    if epoch == (N_EPOCHS - 1): #valid_loss <= best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')        

    #print('WITHOUT ATTENTION: ')
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.4f} | Train PPL: {math.exp(train_loss):7.4f}')
    print(f'\t Val. Loss: {valid_loss:.4f} |  Val. PPL: {math.exp(valid_loss):7.4f}')
    
    # store error value
    train_losses.append(train_loss)
    
# shuffle the dataset and calculate error on training set again
data_train_s = utils.tensor_shuffler(data_train, DEVICE)
start_time = time.time()
prediction_train = evaluate(model, data_train, criterion, DEVICE, datapath)
prediction_train_s = evaluate(model, data_train_s, criterion, DEVICE, datapath)
tt_loss, tt_loss_s = prediction_train['average epoch loss'], prediction_train_s['average epoch loss']
end_time = time.time()
pred_mins, pred_secs = epoch_time(start_time, end_time)

# testing/prediction
#model.load_state_dict(torch.load('tut1-model.pt'))
prediction = evaluate(model, data_test, criterion, DEVICE, datapath)
test_loss = prediction['average epoch loss']

# save predicted values
filename = os.path.join(respath, 'Transformer_test_result_samples.json')
utils.dump_json([prediction['denormalized prediction'][0].tolist(), prediction['denormalized target'][0].tolist()], filename)
filename = os.path.join(respath, 'Transformer_Attention_test_result_samples.json')
utils.dump_json(prediction['attention weights'].tolist(), filename)


print(f'| Test Loss: {test_loss:.4f} | Test PPL: {math.exp(test_loss):7.4f} |')