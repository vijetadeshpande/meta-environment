#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 03:17:28 2020

@author: vijetadeshpande
"""
import gym
import torch.optim as optim
import torch.nn.functional as F
import DDPGAgent as Agent
from DDPGActor import Actor 
from DDPGCritic import Critic
from ReplayBuffer import ReplayBuffer as Buffer

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


def soft_update(estimate, sample):
    
    for param_target, param in zip(estimate.parameters(), sample.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            
    return
   
def main():
    
    # define environment
    env = gym.make(env_name)

    # define agent
    # NOTE: Agent defines actor, critic, target actor/critic, memory buffer ans noise process
    agent = Agent.Agent(env, lr_actor, lr_critic, 
                        buffer_size = buffer_limit, sample_batch_size = batch_size, gamma = gamma, tau = tau)
    
    
    """
    # instantiate the actor and the critic network
    net_actor, net_critic = Actor(env, hidden1_dim, hidden2_dim), Critic(env, hidden1_dim, hidden2_dim)
    net_target_actor, net_target_critic = Actor(env, hidden1_dim, hidden2_dim), Critic(env, hidden1_dim, hidden2_dim)
    
    # instantiate memory/experience buffer
    buffer_memory = Buffer(np.zeros(env.action_space.shape)) 
    
    # define optimizer for actor and critic
    optimizer_actor = optim.Adam(net_actor.parameters(), lr = lr_actor)
    optimizer_critic = optim.Adam(net_critic.parameters(), lr = lr_critic)    
    """
    
    #
    score = 0.0
    print_interval = 20

    for n_epi in range(n_episodes):
        s = env.reset()
        
        for t in range(episode_break):
            # take one step in environment
            #env.render()
            a = agent.actor(s)
            #a = net_actor.forward(s)
            s_next, r, done, info = env.step(a.data)
            
            # save experience
            agent.save_experience(s, a, r, s_next, done)
            """
            buffer_memory.add_experience(s, a, r, s_next, done)
            """
            
            # update score and state
            score +=r
            s = s_next

            if done:
                break              
                
        if agent.buffer.count > 2000:
        #if buffer_memory.count > 2000:
            for i in range(10):
                agent.train()
                
                """
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
                
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()