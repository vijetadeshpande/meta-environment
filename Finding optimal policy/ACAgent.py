#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 23:54:25 2020

@author: vijetadeshpande
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class FFNetwork(nn.Module):
    def __init__(self, learning_rate, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(FFNetwork, self).__init__()
        
        # NN structural attributes
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        
        # define network
        self.layer1 = nn.Linear(self.input_dim, self.hidden_dim1)
        self.layer2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.layer3 = nn.Linear(self.hidden_dim2, self.output_dim)
        self.activation = nn.ReLU()
        
        # define optimizer and scheduler
        self.optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = 300, gamma = 0.5)
        
        # send it to device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, observation):
        
        # convert to torch tensor
        observation = torch.tensor(observation).float().to(self.device)
        
        # forward pass
        x = self.activation(self.layer1(observation))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)
        
        return x

class Agent(object):
    def __init__(self, lr_actor, lr_critic, state_features, output_dim, n_actions,
                 hidden_dim1 = 256, hidden_dim2 = 256, gamma = 0.99):
        
        # attributes common to both actor and critic networks
        self.input_dim = state_features
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        
        # actor network
        self.n_actions = n_actions
        self.params = n_actions * 2
        self.lr_actor = lr_actor
        self.FFactor = FFNetwork(self.lr_actor, self.input_dim, self.hidden_dim1, self.hidden_dim2, self.params)
        
        # critic network
        self.lr_critic = lr_critic
        self.FFcritic = FFNetwork(self.lr_critic, self.input_dim, self.hidden_dim1, self.hidden_dim2, 1)
        
        # NOTE: the output dimension for actior network is equal to (2 * output_features)
        # because we want to predict mean and sd for each action feature. In case
        # of critic network, we are predicting the state value, i.e. one real
        # value for each state. Therefore, the output dim is just 1.
        
        #
        self.gamma = gamma
        self.log_prob = None
        
    def action_selection(self, observation):
        
        # convert to torch tensor
        observation = torch.tensor(observation).float().to(self.FFactor.device)
        
        # 
        param = self.FFactor.forward(observation)
        sd = torch.exp(param[1])
        action_dist = torch.distributions.Normal(param[0], sd)
        action = action_dist.sample(sample_shape = torch.tensor([self.n_actions]))
        self.log_prob = action_dist.log_prob(action).to(self.FFactor.device)
        action = torch.tanh(action)
        
        return [action.item()]
    
    def learn(self, observation, reward, observation_next, done):
        
        # reset the gradients
        self.FFactor.optimizer.zero_grad()
        self.FFcritic.optimizer.zero_grad()
        
        # convert reward to torch tensor
        reward = torch.tensor(reward).float().to(self.FFactor.device)
        
        # predict the state values
        value = self.FFcritic.forward(observation)
        value_next = self.FFcritic.forward(observation_next) if not done else 0
        
        # temporal difference error
        delta = reward + (self.gamma * value_next) - value
        
        #
        critic_loss = delta
        actor_loss = self.log_prob * delta * self.gamma
        (critic_loss + actor_loss).backward()
        self.FFcritic.optimizer.step()
        self.FFactor.optimizer.step()
        
        """
        # critic parameter update
        critic_loss = delta
        critic_loss.backward(retain_graph = True)
        self.FFcritic.optimizer.step()
        
        # actor parameter update
        self.FFactor.optimizer.zero_grad()
        actor_loss = self.log_prob * delta * self.gamma
        actor_loss.backward()
        self.FFactor.optimizer.step()
        
        """
        
        
        
        
    
        