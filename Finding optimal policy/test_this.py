#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 03:17:28 2020

@author: vijetadeshpande
"""
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import DDPGAgent as Agent
#from DDPGActor import Actor 
#from DDPGCritic import Critic
from ReplayBuffer import ReplayBuffer as Buffer
from DDPGNoise import OrnsteinUhlenbeckNoise as Noise

#Hyperparameters
lr_actor        = 0.0005
lr_critic       = 0.001
gamma           = 0.99
batch_size      = 64
buffer_limit    = 50000
tau             = 0.001
n_episodes      = 1000
episode_break   = 300
hidden1_dim     = 64
hidden2_dim     = 32
env_name        = 'Pendulum-v0'


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
        #self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
        
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

class Critic(nn.Module):
    def __init__(self, env, hidden1_dim, hidden2_dim, learning_rate = 0.001):
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
        #self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
        
        # device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, obs, act):
        
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs).float().to(self.device)
        
        h1 = F.relu(self.fc_s(obs))
        h2 = F.relu(self.fc_a(act))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_3(q)
        
        return q


def soft_update(estimate, sample):
    
    for param_target, param in zip(estimate.parameters(), sample.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            
    return
   
def main():
    
    # define environment
    env = gym.make(env_name)
    
    # instantiate the actor and the critic network
    net_actor, net_critic = Actor(env, hidden1_dim, hidden2_dim), Critic(env, hidden1_dim, hidden2_dim)
    net_target_actor, net_target_critic = Actor(env, hidden1_dim, hidden2_dim), Critic(env, hidden1_dim, hidden2_dim)
    
    # instantiate memory/experience buffer
    buffer_memory = Buffer(buffer_limit)
    noise = Noise(np.zeros(env.action_space.shape))
    
    # define optimizer for actor and critic
    optimizer_actor = optim.Adam(net_actor.parameters(), lr = lr_actor)
    optimizer_critic = optim.Adam(net_critic.parameters(), lr = lr_critic)

    """
    # define agent
    # NOTE: Agent defines actor, critic, target actor/critic, memory buffer ans noise process
    agent = Agent.Agent(env, lr_actor, lr_critic, 
                        buffer_size = buffer_limit, sample_batch_size = batch_size, gamma = gamma, tau = tau)
        
    """
    
    #
    score = 0.0
    print_interval = 20

    for n_epi in range(n_episodes):
        s = env.reset()
        
        for t in range(episode_break):
            # take one step in environment
            #env.render()
            #a = agent.actor(s)
            a = net_actor.forward(s) + torch.tensor(noise()).float().to(net_actor.device)
            s_next, r, done, info = env.step(a.data)
            
            # save experience
            #agent.save_experience(s, a, r, s_next, done)
            buffer_memory.add_experience(s, a, r, s_next, done)
        
            
            # update score and state
            score +=r
            s = s_next

            if done:
                break              
                
        #if agent.buffer.count > 2000:
        if buffer_memory.count > 2000:
            for i in range(10):
                
                # sample experiences
                s_b, a_b, r_b, s_next_b, done_b  = buffer_memory.sample_batch()
    
                # calculate td-target via target network
                a_next_b = net_target_actor.forward(s_next_b)
                td_target =  r + (net_target_critic.forward(s_next_b, a_next_b) * gamma)
                
                # calculte td-estimation via critic network
                td_estimation = net_critic(s_b, a_b)
                
                # calculate td-error and update the critic network parameters
                critic_error = F.smooth_l1_loss(td_estimation, td_target.detach())
                optimizer_critic.zero_grad()
                critic_error.backward()
                optimizer_critic.step()
                
                # calculate q value via actor and critic network
                a_hat = net_actor.forward(s_b)
                actor_error = -net_critic.forward(s_b, a_hat).mean()
                optimizer_actor.zero_grad()
                actor_error.backward()
                optimizer_actor.step()
                
                # soft update
                soft_update(net_target_actor, net_actor)
                soft_update(net_target_critic, net_critic)
                
                
                """
                agent.train()
                
                """
                
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()