#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:55:29 2020

@author: vijetadeshpande
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import sys
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data processing, runs generator and utility file')
import HelperFunctions1 as h_fun1


def fill_values(df, time_array, y, feature, model, sample_array, start_idx, end_idx):
    
    # fill values
    df.loc[start_idx:end_idx-1, 't (simulation month)'] = time_array
    df.loc[start_idx:end_idx-1, 'y'] = np.ravel(y, order = 'F')
    df.loc[start_idx:end_idx-1, 'feature'] = feature
    df.loc[start_idx:end_idx-1, 'Model'] = model
    df.loc[start_idx:end_idx-1, 'Example number'] = sample_array

    return df

def filename_to_key(filename):
    
    #
    if 'Van' in filename:
        key = 'RNN Vanilla'
    elif 'GRU' in filename:
        key = 'RNN GRU'
    elif 'LSTM' in filename:
        key = 'RNN LSTM'
    elif ('Attention' in filename) or ('Attn' in filename) or ('attn' in filename):
        key = 'Enc-Dec with attention'
    else:
        key = 'Enc-Dec without attention'
        
    return key
    

class FeatureSequencePlot:
    
    def __init__(self, datapath, plot_title, savepath,
                 palette = 'muted'):
        
        # set attributes
        #self.x_title = x_title
        #self.y_title = y_title
        self.plot_title = plot_title
        self.savepath = savepath
        filename = os.path.join(savepath, plot_title)
        self.filename = filename
        
        # read data
        data = h_fun1.load_all_json(datapath)
        self.data = data
        
        # which test examples to plot
        self.examples = np.arange(2, 5)
        
        # color pallete
        self.col_pal = sns.color_palette(palette)
        
        return

    
    def tensorize(self, data):
        
        data_out = {}
        rand_ex = self.examples
        
        for file in data:
            if len(data[file]) != 2:
                continue
            y_hat, y = data[file][0], data[file][1]
            y_hat, y = np.array(y_hat)[1:, rand_ex, :], np.array(y)[1:, rand_ex, :]
            data_out[filename_to_key(file)] = {'Regression': y_hat, 'CEPAC': y}
        
        return data_out
    
    def reshape_data(self, data):
        
        # take three samples from the test data set
        data = self.tensorize(data)
        
        # shape parameters
        NUM_MODELS = len(data)
        TRG_SEQ, TEST_SAMPLES, OUT_FEATURES = data['RNN Vanilla']['CEPAC'].shape
        
        # initiate dataframe to store the results
        df = pd.DataFrame(-1, index = np.arange(TRG_SEQ * NUM_MODELS * TEST_SAMPLES * OUT_FEATURES), columns = ['t (simulation month)', 'y', 'feature', 'Model', 'Example number'])
        df_cepac = pd.DataFrame(-1, index = np.arange(TRG_SEQ * TEST_SAMPLES * OUT_FEATURES), columns = ['t (simulation month)', 'y', 'feature', 'Model', 'Example number'])
        
        # initiation
        start_idx, step, end_idx = 0, (TRG_SEQ*TEST_SAMPLES), 0+(TRG_SEQ*TEST_SAMPLES)
        sample_array = []
        for example in range(TEST_SAMPLES):
            for t in range(TRG_SEQ):
                sample_array.append(example)
        time_array = [i for i in range(TRG_SEQ)] * TEST_SAMPLES
        
        # iterate over results from different models
        for model in data:
            
            # fill values in df for plotting
            feature_idx = -1
            for feature in ['trans.', 'infec.', 'sus.']:
                feature_idx += 1
                # fill predicted values
                df = fill_values(df, time_array, data[model]['Regression'][:, :, feature_idx], feature, model, sample_array, start_idx, end_idx)
                # update row
                start_idx, end_idx = start_idx + step, end_idx + step
        
        # save values for CEPAC
        start_idx, step, end_idx = 0, (TRG_SEQ*TEST_SAMPLES), 0+(TRG_SEQ*TEST_SAMPLES)
        feature_idx = -1
        for feature in ['trans.', 'infec.', 'sus.']:
            feature_idx += 1
            # fill predicted values
            df_cepac = fill_values(df_cepac, time_array, data[model]['CEPAC'][:, :, feature_idx], feature, 'CEPAC', sample_array, start_idx, end_idx)
            # update row
            start_idx, end_idx = start_idx + step, end_idx + step
        
        # join df and df_cepac for plotting
        df = pd.concat([df, df_cepac])
        
        
        return df
    
    def plot_and_save(self, df, example, feature):
        
        #
        plt.figure()
        sns.lineplot(data = df,
                     x = 't (simulation month)',
                     y = 'y',
                     hue = 'Model')
        
        #
        filename = os.path.join(self.savepath, r'Results for example no. %d and feature = %s'%(example, feature))
        plt.savefig(filename, dpi = 360)
        
        return
        
    def save_results(self):
        
        # convert data into reqquired format
        df = self.reshape_data(self.data)
        
        # df should have following columns
        # 1. Total number of infections
        # 2. Total number of transmission
        # 3. Total number of susceptible
        # 4. t (simulation month)
        # 5. RNN model
        # 6. Sample test example
        
        # set plot encironment
        sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        sns.set_context("notebook", rc={"lines.linewidth": 1.5}, font_scale = 1.2)
        aspect_r = 1.4
        line_alpha = 0.9
        
        
        # try plotting just one plot 
        plt.figure()
        g = sns.FacetGrid(data = df, 
                         row = 'Example number',
                         col = 'feature',
                         sharex = False,
                         sharey = False,
                         aspect = aspect_r)#, hue="Coverage time")#, col_wrap=3)
        g = (g.map(sns.lineplot, 
                   "t (simulation month)", 
                   'y',
                   'Model',
                   alpha = line_alpha).add_legend())#, "WellID")
        
        # title
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle('Prediction from RNN models')
        
        # save
        filename = os.path.join(self.savepath, r'Results from RNN models')
        plt.savefig(filename, dpi = 720)
        
        """
        # 
        for example in range(len(self.examples)):
            for feature in ['trans.', 'infec.', 'sus.']:
                float_df = df.loc[df.loc[:, 'feature'] == feature, :]
                float_df = float_df.loc[float_df.loc[:, 'Example number'] == example, :]
                self.plot_and_save(float_df, example, feature)
        """
        
class ErrorPlot:
    
    def __init__(self, datapath, plot_title, savepath,
                 palette = 'muted'):
        
        # set attributes
        #self.x_title = x_title
        #self.y_title = y_title
        self.plot_title = plot_title
        self.savepath = savepath
        filename = os.path.join(savepath, plot_title)
        self.filename = filename
        
        # read data
        data = h_fun1.load_all_csv(datapath)
        self.data = self.reshape_data(data)
        
        # which test examples to plot
        self.examples = np.arange(2, 5)
        
        # color pallete
        self.col_pal = sns.color_palette(palette)
        
        return
    
    def reshape_data(self, data_in):
        
        files = list(data_in.keys())
        N_FILES, N_EPOCHS, N_TRIALS = len(files), len(data_in[files[0]].index), len(data_in[files[0]].columns) - 2
        ROWS = N_FILES * N_EPOCHS * N_TRIALS
        data_out = pd.DataFrame(-1, index = np.arange(ROWS), columns = ['Epoch', 'Mean Squared Error', 'Trial', 'Model'])
        error_columns = []
        for i in list(data_in[files[0]].columns):
            if 'Error' in i:
                error_columns.append(i)
        trial_array = np.ravel(np.repeat(np.reshape(np.arange(N_TRIALS), (N_TRIALS, 1)), N_EPOCHS, axis = 1), order = 'C')
        
        #
        idx_row = 0
        STEP = N_TRIALS * N_EPOCHS
        for file in data_in:
            data_out.loc[idx_row: idx_row + STEP - 1, 'Epoch'] = np.repeat(np.arange(N_EPOCHS), N_TRIALS)
            data_out.loc[idx_row: idx_row + STEP - 1, 'Mean Squared Error'] = np.reshape(data_in[file].loc[:, error_columns].to_numpy(), (STEP,))
            data_out.loc[idx_row: idx_row + STEP - 1, 'Trial'] = trial_array
            data_out.loc[idx_row: idx_row + STEP - 1, 'Model'] = filename_to_key(file)
            idx_row += STEP
            
        return data_out
    
    def plot_and_save(self):
        
        # get data
        data = self.data
        
        # set plot encironment
        sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        sns.set_context("notebook", rc={"lines.linewidth": 1.5}, font_scale = 1.2)
        aspect_r = 2
        line_alpha = 0.9
        
        # plot and save
        plt.figure(figsize=(8, 6))
        sns.lineplot(data = data,
                     x = 'Epoch',
                     y = 'Mean Squared Error',
                     hue = 'Model',
                     ci = 'sd')
        plt.savefig(self.filename, dpi = 360)
    
        
        