#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 02:29:13 2020

@author: vijetadeshpande
"""

import numpy as np
import torch
import os
import pandas as pd
import seaborn as sns
import PGAgent as Agent
import matplotlib.pyplot as plt
import gym


# create environment
env = gym.make('LunarLander-v2')
agent = Agent.Agent(learning_rate = 0.001, input_dim = [8], n_actions = 4)
scores, avg_scores = [], []
score = 0
n_episodes = 500

#
env.render()
for i in range(n_episodes):
    score = 0
    done = False
    observation = env.reset()
    while not done:
        #env.render()
        action = agent.action_selection(observation)
        observation_next, reward, done, info = env.step(action)
        agent.reward_memory.append(reward)
        observation = observation_next
        score += reward
    scores.append(score)
    avg_scores.append(np.mean(scores[-100:]))
    agent.FFpolicy.scheduler.step()
    
    # call the learning function here
    agent.learn()
    
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
