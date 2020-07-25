#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 03:17:28 2020

@author: vijetadeshpande
"""
import gym
import DDPGAgent as Agent
from DDPGActor import Actor 
from DDPGCritic import Critic

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
   
def main():
    
    # define environment
    env = gym.make(env_name)

    # define agent
    # NOTE: Agent defines actor, critic, target actor/critic, memory buffer ans noise process
    agent = Agent.Agent(env, lr_actor, lr_critic, 
                        buffer_size = buffer_limit, sample_batch_size = batch_size, gamma = gamma, tau = tau)
    
    # instantiate the actor and the critic network
    #net_actor, net_critic = Actor(env, hidden1_dim, hidden2_dim), Critic
    
    #
    score = 0.0
    print_interval = 20

    for n_epi in range(n_episodes):
        s = env.reset()
        
        for t in range(episode_break):
            #env.render()
            a = agent.actor(s)
            s_next, r, done, info = env.step(a.data)
            agent.save_experience(s, a, r, s_next, done)
            score +=r
            s = s_next

            if done:
                break              
                
        if agent.buffer.count > 2000:
            for i in range(10):
                agent.train()
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()