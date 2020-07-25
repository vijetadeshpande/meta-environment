#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:35:11 2020

@author: vijetadeshpande
"""

import gym
import QAgent as Agent
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

env = gym.make('LunarLander-v2')
agent = Agent.Agent(gamma = 0.98, epsilon = 1, learning_rate = 0.002,
                    input_dims = [8], batch_size = 100, n_actions = 4)
returns, epsilons, avg_returns = [], [], []
max_games = 1000

for i in range(max_games):
    score = 0
    done = False
    observation = env.reset()
    while not done:
        env.render()
        action = agent.action_selection(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        agent.store_transition(observation, action, reward, observation_, done)
        agent.learn()
        observation = observation_
    
    # store
    epsilons.append(agent.epsilon)
    returns.append(score)
    avg_return = np.mean(returns[-100:])
    avg_returns.append(avg_return)
    
    # scheduler step
    agent.FFvalues.scheduler.step()
        
    # print progress
    print('Episode no:', i, 'Score %.4f' % score, 'Average score %.4f' % avg_return, 
          'Epsilon %.4f' % agent.epsilon)
        
# plot and save
filename = 'LunarLander_.png'
plot_df = pd.DataFrame(0, index = np.arange(max_games), columns = ['x', 'Scores'])
plot_df['Episode'] = np.arange(max_games)
plot_df['Return'] = avg_returns
plt.figure()
sns.lineplot(data = plot_df, 
             x = 'Episode', 
             y = 'Return')
plt.savefig(filename)