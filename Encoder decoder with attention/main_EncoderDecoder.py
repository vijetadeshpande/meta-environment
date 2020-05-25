#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:28:10 2020

@author: vijetadeshpande
"""

from ModelData import ModelData
from Encoder import Encoder
from Decoder import Decoder
from Seq2Seq import Seq2Seq
from Attention import Attention as Att
from Attention import CosineAttention as CosAtt
from train_EncDec import train
from evaluate_EncDec import evaluate
import time
import torch
import torch.nn as nn
import torch.optim as optim
import os
import math
import sys
sys.path.insert(1, r'/Users/vijetadeshpanlde/Documents/GitHub/meta-environment/Data processing, runs generator and utility file')
import utils
import seaborn as sns
from copy import deepcopy

# path variables
datapath = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/CEPAC RUNS/regression model input'
respath = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/RNN results'

# create data object
data_object = ModelData(datapath, batch_size = 128)
data_train, data_test = data_object.train_examples, data_object.test_examples

# parameters for defining encoder and decoder
INPUT_DIM, OUTPUT_DIM = data_object.input_features, data_object.output_features
ENC_HID_DIM = 64
DEC_HID_DIM = 64
N_LAYERS = 1
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
DEVICE = 'cpu'

# initialize encoder, decoder and seq2seq model classes
enc = Encoder(INPUT_DIM, ENC_HID_DIM, N_LAYERS, ENC_DROPOUT, is_bidirectional = False)
attn = CosAtt(enc, DEC_HID_DIM, N_LAYERS)
dec = Decoder(OUTPUT_DIM, DEC_HID_DIM, N_LAYERS, DEC_DROPOUT, enc, attention = attn)
model = Seq2Seq(enc, dec, DEVICE, attention = attn)

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
criterion = nn.MSELoss() #nn.SmoothL1Loss()

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
for epoch in range(N_EPOCHS):
    
    # WITHOUT ATTENTION
    # start clock
    start_time = time.time()
    # train
    train_loss, _ = train(model, data_train, optimizer, criterion, CLIP, DEVICE)
    # stop clock
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    # print
    # if we happen to have a validation data set then calculate validation
    # loss here by predicting the value of validation set 'x's
    valid_loss = 0
    # update validation loss if less than previously observed minimum
    if valid_loss <= best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')        

    #print('WITHOUT ATTENTION: ')
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.4f} | Train PPL: {math.exp(train_loss):7.4f}')
    print(f'\t Val. Loss: {valid_loss:.4f} |  Val. PPL: {math.exp(valid_loss):7.4f}')

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
filename = os.path.join(respath, 'EncoderDecoder_test_result_samples.json')
utils.dump_json([prediction['denormalized prediction'][0].tolist(), prediction['denormalized target'][0].tolist()], filename)
filename = os.path.join(respath, 'EncoderDecoder_Attention_test_result_samples.json')
utils.dump_json(prediction['attention weights'].tolist(), filename)


print(f'| Test Loss: {test_loss:.4f} | Test PPL: {math.exp(test_loss):7.4f} |')

# plot attention w
attention_w = deepcopy(prediction['attention weights'])
aaa = attention_w[0].detach().numpy()
sns.heatmap(aaa[0, 1:, :])

#x = r'/Users/vijetadeshpande/Documents/GitHub/Sequence2Sequence model for CEPAC prediction/test check/results'
#link.export_output_to_excel(x, x)
