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
import torch
import pandas as pd
from copy import deepcopy
import argparse


class Space:
    def __init__(self, shape, low, high):
        self.shape = shape
        self.high = high
        self.low = low
    
    def sample(self):
        
        return np.random.uniform(low = self.low[0], high = self.high[0], size = self.shape[0])
        

class Environment:
    def __init__(self, RNN_model = 'Vanilla', 
                 parameter_path = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/CEPAC RUNS/regression model input'):
        if RNN_model == 'Vanilla':
            self.model = FVan(r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Hyper parameter tuning/trained model parameters/RNNVan.pt')
        elif RNN_model == 'GRU':
            self.model = FGRU()
        elif RNN_model == 'LSTM':
            self.model = FLSTM()
        else:
            self.model = None
            
        # what is the budget
        #self.budget = budget # it should be a number from 0 to 1, 1 corresponds resource enough to avert 10000 infections
        
        # observation and action space
        self.observation_space = self.init_observation_space()
        self.action_space = self.init_action_space()
        
        # 
        self.z_path = parameter_path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # attribute to store SQ outcomes
        self.sq_outcomes = []
        
        # population size of each city
        self.pop_size = [94105, 94105, 94105]
        self.cohort_size = 10000000
        self.cost_per_person = 1
        self.resource_bound = 10000
        self._max_episode_steps = 2
        
        
    def init_observation_space(self):
        
        #
        state_feature_rep = pd.read_csv(r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/CEPAC RUNS/regression model input/MDP_state.csv')
        state_feature_rep = state_feature_rep.set_index('city').to_numpy()
        state_feature_rep = np.clip(state_feature_rep, a_min = -2, a_max = 2)
        features = state_feature_rep.shape[0] * state_feature_rep.shape[1]
        state_feature_rep = np.reshape(state_feature_rep, (features), order = 'F')
        
        # we have all epidemic features, only thing we need now is a feature
        # representing total resources
        start_state = np.zeros(len(state_feature_rep) + 1)
        start_state[0] = np.random.uniform()
        start_state[1:] = state_feature_rep
        
        # NOTE: At this point we shoud have a state representation of the MDP, 
        # which includes mainly two types of features. One feature (first value)
        # representing how much is the budget. Note that this feature value
        # goes from 0 to 1, 1 corresponds resource enough to avert 10000 infections.
        # rest of the features are for epidemic properties.
        self.start_state = start_state
        
        # low
        low = -2 * np.ones(len(start_state))
        low[0] = 0
        low[10:22] = 0
        
        # high
        high = 2 * np.ones(len(start_state))
        high[0] = 1
        high[10:22] = 1
        
        #
        observation_space = Space((len(start_state), ), low, high)
        
        return observation_space
    
    def init_action_space(self):
        
        # to get the number of cities
        state_feature_rep = pd.read_csv(r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/CEPAC RUNS/regression model input/MDP_state.csv')
        cities = state_feature_rep.shape[0]
        self.cities = cities
        
        # low and high
        low = -1 * np.ones(cities)
        high = 1 * np.ones(cities)
        
        #
        action_space = Space((cities * 2, ), low, high)
        
        return action_space
        
    def forward(self, input_signal):
        
        X, Y = input_signal[0][0], input_signal[0][1]
        
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X).float()
        if not isinstance(Y, torch.Tensor):
            BATCH, SRC_LEN, _ = X.shape
            Y = torch.zeros((BATCH, SRC_LEN+1, self.cities)).float()
        
        input_signal = [[X, Y]]
        
        prediction = self.model.forward(input_signal, self.z_path, self.device)
        
        return prediction['denormalized prediction'][0]
    
    def community_benefit(self, action):
        
        sq_outcomes = self.sq_outcomes
        if isinstance(sq_outcomes, list):
            # Status Quo
            sq_action = {'PrEP enable': [0, 0, 0], 'PrEP coverage': action[:3], 
                         'PrEP duration': action[3:], 'PrEP shape': [2, 2, 2]}
            sq_input_signal = [[featurization(sq_action), []]]
            sq_outcomes = self.forward(sq_input_signal)
            self.sq_outcomes = sq_outcomes
        
        # intervention
        int_action = {'PrEP enable': [1, 1, 1], 'PrEP coverage': action[:3], 
                      'PrEP duration': action[3:], 'PrEP shape': [2, 2, 2]}
        int_input_signal = [[featurization(int_action), []]]
        int_outcomes = self.forward(int_input_signal)
        
        # community benefit
        coefficients = []
        for city in range(3):
            percentage, coefficient = h_fun2.community_benefit(sq_outcomes[:, city, :], int_outcomes[:, city, :])
            coefficients.append(coefficient)
        
        # create output as required for forward pass of main run
        c_benefit = [[1, 1, 1], [60, 60, 60], coefficients]
        
        return c_benefit
    
    def denormalize_action(self, action):
        
        # action from the RL code will be in between -1 to 1 (bacuase of tanh)
        # we use min-max scaler to shift the values between 0 to 1.
        # The assuming the [0, 1, 2] feature to be pwercentage coverage,
        # we multiply those features by 1 and other features by 60 as the other
        # features will represent the time to reach the maximum uptake
        action = (action - (-1))/(1 - (-1))
        
        # scale percentage uptake
        action[:3] *= 1
        
        # scale time to reach max uptake
        action[3:] *= 60
        
        return action
    
    def meta_simulation(self, action):
        
        # denormalize action
        action = self.denormalize_action(action)
        
        # calculate community benefit for current action
        c_benefit = self.community_benefit(action)
        
        # restructure action
        action = {'PrEP enable': [1, 1, 1], 'PrEP coverage': action[:3], 
                  'PrEP duration': action[3:], 'PrEP shape': [2, 2, 2]}
        
        # featurization
        input_signal = featurization(action, c_benefit)
        input_signal = [[input_signal, []]]
        
        # meta-simulation
        outcomes = self.forward(input_signal)
        
        return outcomes
    
    def calculate_averted_infections(self, action):
        
        # meta simulation
        int_outcomes = self.meta_simulation(action)
        
        # get sq outcomes
        sq_outcomes = self.sq_outcomes
        
        # 
        infections_averted = (sq_outcomes - int_outcomes)[:, :, 1]
        infections_averted = np.sum(infections_averted, axis = 0)
        infections_averted = np.multiply(np.divide(infections_averted, self.cohort_size), self.pop_size)
        #infections_averted = np.sum(np.mutiply(infections_averted, self.cost_per_person))
        
        return infections_averted
    
    def reset(self):
        
        # TODO: not sure how to handle the following hard coding
        
        # import feature representation of start state of MDP
        feature_rep = self.start_state #(pd.read_csv(r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/CEPAC RUNS/regression model input/MDP_state.csv')).values
        
        # take random sample for the resource feature value
        # it means that you start in some ramdom state eachtime you reset
        feature_rep[0] = np.random.uniform()
        self.start_state = feature_rep
        
        return feature_rep
        
        
    def step(self, action):
        
        # state
        state = deepcopy(self.start_state)
        
        # reward
        averted_infections = self.calculate_averted_infections(action)
        if averted_infections.sum() > np.multiply(self.resource_bound, state[0]):
            reward = -1 * (averted_infections.sum() - np.multiply(self.resource_bound, state[0]))
        else:
            reward = averted_infections.sum()
        
        # next state is always terminal
        s_next = deepcopy(state)
        s_next[[2, 11, 20]] = 0 # terminal state incidence
        s_next[[7, 16, 25]] = 0 # terminal state prep incidence
        self.start_state = s_next
        
        #
        done = True
        info = {}
        
        # 
        if reward == float('NaN'):
            print(state)
            print(action)
        
        return (s_next, reward, done, info)
        


#env = Environment()
#s = env.reset()
#a = np.random.uniform(size = env.action_space.shape)
#s_next, r, done, info = env.step(a)
        
        