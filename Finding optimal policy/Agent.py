#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 00:28:08 2020

@author: vijetadeshpande
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class FFBellmanValues(nn.Module):
    def __init__(self, learning_rate, input_dim, stepdown1_dim, n_actions):
        super(FFBellmanValues, self).__init__()
        
        # structural attributes for the ff network
        self.input_dim = input_dim
        self.stepdown1 = stepdown1_dim
        self.output_dim = n_actions
        
        # define ff network (not using stepdown right now)
        self.network = nn.Linear(self.input_dim, self.output_dim)
        
        # optimizer and criterion definition
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        self.criterion = nn.MSELoss()
        
        # send everything to device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        
        # not using actiovation here
        action_values = self.network(state)
        
        return action_values
    
class Agent():
    def __init__(self, gamma, epsilon, learning_rate, input_dims, batch_size,
                 n_actions, max_mem_size = 10000, epsilon_min = 0.01, epsilon_reduction = 5e-4):
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_reduction = epsilon_reduction
        self.learning_rate = learning_rate
        self.action_space = [i for i in range(n_actions)]
        self.memory_size = max_mem_size
        self.batch_size = batch_size
        self.memory_cntr = 0
        
        # define FFN for storing the values
        self.FFvalues = FFBellmanValues(self.learnig_rate, self.input_dim, 32, n_actions)
        
        # store the memory
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtyoe = np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype = np.float32)
        self.action_memory = np.zeros((self.mem_size, *input_dims), dtype = np.float32)
        self.reward_memory = np.zeros((self.mem_size, *input_dims), dtype = np.float32)
        self.terminal_memory = np.zeros((self.mem_size, *input_dims), dtype = np.float32)
        
    def store_transition(self, state_cur, action_cur, reward, state_next, done):
        index = self.memory_cntr % self.memory_size
        self.state_memory[index] = state_cur
        self.action_memory[index] = action_cur
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_next
        self.terminal_memory[index] = done
        
        self.memory_cntr += 1
    
    def action_selection(self, observation):
        
        # epsilon greedy action selection
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation]).to(self.Q_eval.device)
            action_values = self.FFvalues(state)
            action = torch.argmax(action_values).item()
        else:
            action = np.random.choice(self.action_space)
            
        
        return action
    
    def learn(self):
        
        #
        if self.mem_cntr < self.batch_size:
            return
        
        # start learning by setting the gradients to zero
        self.FFvalues.optimizer.zero_grad()
        
        # 
        max_mem = min(self.memory_cntr, self.memory_size)
        batch = np.random.choice(max_mem, self.batch_size, replace = False)
        batch_index = np.arange(self.batch_size, dtype = np.float32)
        
        # access the required S-A-R-S
        state_batch = torch.tensor(self.state_memory[batch]).to(self.FFvalues.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.FFvalues.device)
        action_batch = self.action_memory[batch]
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.FFvalues.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.FFvalues.device)
        
        # evalute the Bellman values
        q_cur = self.FFvalues(state_batch)[batch_index, action_batch]
        q_next = self.FFvalues(new_state_batch)
        q_next[terminal_batch] = 0
        q_target = reward_batch + self.gamma * torch.max(q_next, dim = 1)[0]
        
        # calculate loss
        loss = self.FFvalues.criterion(q_target, q_cur).to(self.FFvalue.device)
        loss.backwards()
        
        # update parameters
        self.FFvalues.optimizer.step()
        
        # update the exploration hyper-par
        self.epsilon = self.epsilon - self.epsilon_reduction if self.epsilon > self.epsilon_min else self.epsilon_min
        
        
        
        
        
        
        
        
        
    