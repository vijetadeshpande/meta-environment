#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 02:07:37 2020

@author: vijetadeshpande
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Critic(nn.Module):
    def __init__(self, env, hidden1_dim, hidden2_dim, learning_rate):
        super(Critic, self).__init__()
        
        # 
        self.observation_features = env.observation_space.shape[0]
        self.action_features = env.action_space.shape[0]
        self.lr = learning_rate
        
        # define network
        self.fc_s = nn.Linear(self.observation_features, hidden1_dim)
        self.fc_a = nn.Linear(self.action_features, hidden1_dim)
        self.fc_q = nn.Linear(hidden1_dim * 2, hidden2_dim)
        self.fc_3 = nn.Linear(hidden2_dim, 1) # 1 because there will only be on Q value for each (s, a) pair
        
        # define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
        
        # device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, obs, act):
        
        h1 = F.relu(self.fc_s(obs))
        h2 = F.relu(self.fc_a(act))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_3(q)
        
        return q