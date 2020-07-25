#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 01:12:23 2020

@author: vijetadeshpande
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os


class PolicyNetwork(nn.Module):
    def __init__(self, learning_rate, input_dim, hidden_dim1, hidden_dim2, n_actions):
        super(PolicyNetwork, self).__init__()
        
        # save structural attributes
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = n_actions
        
        # NOTE: here, our actions space can be continuous, hence, think of
        # output dim as number of features defining an action in action space
        
        # define the network
        self.network1 = nn.Linear(*self.input_dim, self.hidden_dim1)
        self.network2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.network3 = nn.Linear(self.hidden_dim2, self.output_dim)
        self.activation = nn.ReLU()
        
        # define optimizer, criterion and scheduler
        self.optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)
        #self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = 300, gamma = 0.5)
        
        # define device send everything to device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, observation):
        observation = torch.tensor(observation).float().to(self.device)
        action_preference = self.activation(self.network1(observation))
        action_preference = self.activation(self.network2(action_preference))
        action_preference = self.network3(action_preference)
        
        return action_preference
    
class Agent():
    def __init__(self, learning_rate, input_dim, n_actions, 
                 gamma = 0.99, hidden_dim1 = 512, hidden_dim2 = 512, memory_size = 100000):
        
        # define the policy network
        self.FFpolicy = PolicyNetwork(learning_rate, input_dim, hidden_dim1, hidden_dim2, n_actions)
        
        #
        self.gamma = gamma  
        self.memory_size = memory_size
        
        # memory space
        self.action_memory = []#np.zeros(self.memory_size, dtype = np.float32)
        self.reward_memory = []#np.zeros(self.memory_size, dtype = np.float32)
    
    def store_transition(self, reward):
        self.reward_memory.append(reward)
        
        return
    
    def action_selection(self, observation):
        
        #
        action_distribution = F.softmax(self.FFpolicy.forward(observation), dim = 0)
        action_probs = torch.distributions.Categorical(action_distribution)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        
        # save action in memory
        self.action_memory.append(log_prob)
        
        return action.item()
    
    def learn(self):
        
        # make gradients equal to zero
        self.FFpolicy.optimizer.zero_grad()
        
        # get the returns for the episode
        G = np.zeros_like(self.reward_memory, dtype = np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += discount * self.reward_memory[t]
                discount *= self.gamma
            
            G[t] = G_sum 
            
        # standardize the G values
        mean = np.mean(G)
        sd = np.std(G) if np.std(G) > 0 else 1
        G = (G - mean)/sd
        G = torch.tensor(G).float().to(self.FFpolicy.device)
        
        # calculate loss
        loss = 0
        for g, logprob in zip(G, self.action_memory):
            loss += (-g * logprob)
            
        # backpropogation and parameter updation
        loss.backward()
        self.FFpolicy.optimizer.step()
        
        # reset the reward and action memory
        self.action_memory = []
        self.reward_memory = []
        
        return
            
    
        