#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 01:49:40 2020

@author: vijetadeshpande
"""
import numpy as np
import torch
import os
import pandas as pd
import seaborn as sns
import ACAgent as Agent
import matplotlib.pyplot as plt
import gym

# create environment
env = gym.make('MountainCarContinuous-v0')
agent = Agent.Agent(lr_actor = 0.001, lr_critic = 0.001, 
                    state_features = 2, output_dim = 1, n_actions = 1)
scores, avg_scores = [], []
score = 0
n_episodes = 100

#
for i in range(n_episodes):
    score = 0
    done = False
    observation = env.reset()
    
    while not done:
        #env.render()
        action = agent.action_selection(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        agent.learn(observation, reward, observation_, done)
        observation = observation_
    
    #
    scores.append(score)
    avg_scores.append(np.mean(scores[-100:]))
    agent.FFactor.scheduler.step()
    agent.FFcritic.scheduler.step()
    
    # print progress
    print('Episode no:', i, 'Score %.4f' % score, 'Average score %.4f' % avg_scores[i])
    
# plot and save
filename = 'MountainCarConti_.png'
plot_df = pd.DataFrame(0, index = np.arange(n_episodes), columns = ['x', 'Scores'])
plot_df['Episode'] = np.arange(n_episodes)
plot_df['Return'] = avg_scores
plt.figure()
sns.lineplot(data = plot_df, 
             x = 'Episode', 
             y = 'Return')
plt.savefig(filename)