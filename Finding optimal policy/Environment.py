#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 15:36:20 2020

@author: vijetadeshpande
"""
import sys
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data processing, runs generator and utility file') 
from action_to_input_signal import get_featurized_input as featurization
from FunctionApproximator import GRUApproximator as FGRU
from FunctionApproximator import TransformerApproximator as FTrans
from FunctionApproximator import LSTMApproximator as FLSTM
from FunctionApproximator import VanillaApproximator as FVan
import HelperFunctions2 as h_fun2
import numpy as np

class Environment:
    def __init__(self, RNN_model = 'Vanilla', 
                 parameter_path = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/CEPAC RUNS/regression model input',
                 device = 'cpu'):
        if RNN_model == 'Vanilla':
            self.model = FVan()
        elif RNN_model == 'GRU':
            self.model = FGRU()
        elif RNN_model == 'LSTM':
            self.model = FLSTM()
        else:
            self.model = None
        
        #
        self.z_path = parameter_path
        self.device = device
        
        # attribute to store SQ outcomes
        self.sq_outcomes = None
        
        # population size of each city
        self.pop_size = [94105, 94105, 94105]
        self.cohort_size = 10000000
        self.cost_per_person = 1
        
        
    def forward(self, input_signal):
        
        prediction = self.RNN_model.forward(input_signal, self.z_path, self.device)
        
        return prediction['Denormalized prediction']
    
    def community_benefit(self, action):
        
        # Status Quo
        sq_action = {'PrEP enable': [0, 0, 0], 'PrEP coverage': action[1], 
                     'PrEP duration': action[2], 'PrEP shape': action[3]}
        sq_input_signal = [[featurization(sq_action), []]]
        sq_outcomes = self.forward(sq_input_signal)
        if self.sq_outcomes == None:
            self.sq_outcomes = sq_outcomes
        
        # intervention
        int_action = {'PrEP enable': action[0], 'PrEP coverage': action[1], 
                      'PrEP duration': action[2], 'PrEP shape': action[3]}
        int_input_signal = [[featurization(int_action), []]]
        int_outcomes = self.forward(int_input_signal)
        
        # community benefit
        coefficients = []
        for city in range(len(action[0])):
            percentage, coefficient = h_fun2.community_benefit(sq_outcomes[:, city, :], int_outcomes[:, city, :])
            coefficients.append(coefficient)
        
        # create output as required for forward pass of main run
        c_benefit = [[1, 1, 1], [60, 60, 60], coefficients]
        
        return c_benefit
    
    def meta_simulation(self, action):
        
        # calculate community benefit for current action
        c_benefit = self.community_benefit(action)
        
        # restructure action
        action = {'PrEP enable': action[0], 'PrEP coverage': action[1], 
                  'PrEP duration': action[2], 'PrEP shape': action[3]}
        
        # featurization
        input_signal = featurization(action, c_benefit)
        input_signal = [[input_signal, []]]
        
        # meta-simulation
        outcomes = self.forward(input_signal)
        
        return outcomes
    
    def calculate_reward(self, action):
        
        # meta simulation
        int_outcomes = self.meta_simulation(action)
        
        # get sq outcomes
        sq_outcomes = self.sq_outcomes
        
        # 
        infections_averted = (sq_outcomes - int_outcomes)[:, :, 1]
        infections_averted = np.sum(infections_averted, axis = 0)
        infections_averted = np.multiply(np.divide(infections_averted, self.cohort_size), self.pop_size)
        infections_averted = np.sum(np.mutiply(infections_averted, self.cost_per_person))
        
        return infections_averted
        
        
        