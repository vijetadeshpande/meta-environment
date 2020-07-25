#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 01:58:40 2020

@author: vijetadeshpande
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, env, hidden1_dim, hidden2_dim, learning_rate = 0.001):
        super(Actor, self).__init__()
        
        #
        self.action_space = env.action_space
        self.action_features = env.action_space.shape[0]
        self.observation_features = env.observation_space.shape[0]
        self.lr = learning_rate
        
        # define network
        self.fc1 = nn.Linear(self.observation_features, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.fc_mu = nn.Linear(hidden2_dim, self.action_features)
        
        # define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
        
        # device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, obs):
        
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs).float().to(self.device)
        
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc_mu(x))
        u = torch.tensor(self.action_space.high).float().to(self.device)
        action = torch.mul(action, u)
        
        return action