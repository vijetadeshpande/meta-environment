#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 02:18:46 2020

@author: vijetadeshpande
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from DDPGActor import Actor as Actor
from DDPGCritic import Critic as Critic
from DDPGNoise import OrnsteinUhlenbeckNoise as Noise
from ReplayBuffer import ReplayBuffer

class Agent:
    def __init__(self, env, lr_actor, lr_critic, hidden1_dim = 64, hidden2_dim = 32, 
                 buffer_size = 3000, sample_batch_size = 64, gamma = 0.97, tau = 0.01):
        
        # Actor and critic networks
        self.actor = Actor(env, hidden1_dim, hidden2_dim, lr_actor)
        self.critic = Critic(env, hidden1_dim, hidden2_dim, lr_critic)
        
        # target actor and target critic networks
        self.target_actor = Actor(env, hidden1_dim, hidden2_dim, lr_actor)
        self.target_critic = Critic(env, hidden1_dim, hidden2_dim, lr_critic)
        
        # initialize weights
        # TODO: add code for weight initialization for actor and critic net
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # define noise
        mu = np.zeros(env.action_space.shape)
        self.noise = Noise(mu)
        
        # replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        #
        self.sample_batch = sample_batch_size
        self.gamma = gamma
        self.tau = tau
        
    def take_action(self, observation):
        
        if not isinstance(observation, torch.Tensor):
            observation = torch.from_numpy(observation).float().device(self.actor.device)
            
        action = self.actor.forward(observation) + torch.tensor(self.noise()).float().to(self.actor.device)
        #action = action.numpy()
        
        return action
    
    def save_experience(self, s, a, r, s_next, done):
        
        self.buffer.add_experience(s, a, r, s_next, done)
        
        return
    
    def soft_update(self):
        
        for param_target, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
            
        for param_target, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
    
    def train(self):
        
        obs, act, r, obs_next, done  = self.buffer.sample_batch()
    
        # calculate td-target via target network
        act_next = self.target_actor.forward(obs_next)
        td_target =  r + (self.target_critic.forward(obs_next, act_next) * self.gamma * done)
        
        # calculte td-estimation via critic network
        td_estimation = self.critic(obs, act)
        
        # calculate td-error and update the critic network parameters
        self.actor.eval()
        critic_error = F.smooth_l1_loss(td_estimation, td_target.detach())
        self.critic.optimizer.zero_grad()
        critic_error.backward()
        self.critic.optimizer.step()
        
        # calculate q value via actor and critic network
        self.actor.train()
        a_hat = self.take_action(obs)
        actor_error = -self.critic.forward(obs, a_hat).mean()
        self.actor.optimizer.zero_grad()
        actor_error.backward()
        self.actor.optimizer.step()
        
        # soft update
        self.soft_update()
        
        return critic_error.detach().numpy(), actor_error.detach().numpy()
        
        
        