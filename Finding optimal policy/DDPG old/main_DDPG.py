import gym
import DDPGAgent as Agent
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# define the environment
env = gym.make('Pendulum-v0') #gym.make('LunarLanderContinuous-v2')  #gym.make('Pendulum-v0')

# create instance of the agent class
agent = Agent.Agent(env, lr_actor = 0.0005, lr_critic = 0.001)

# initialize lists to store rewards and actor and critic error
returns, avg_returns, actor_errors, critic_errors = [], [], [], []
n_episodes = 3000

# start primary training loop over all the episodex
for i in range(n_episodes):
    score = 0
    done = False
    observation = env.reset()
    while not done:
        #env.render()

        # take action (via actor network) and make a step in env
        action = agent.action_selection(observation)
        observation_, reward, done, info = env.step(action)

        # save the experience
        agent.save_experience(observation, action, [reward], observation_, [done])

        # update score
        score += reward

        # update the current state
        observation = observation_
    
    # update the network parameters
    actor_loss, critic_loss = 0, 0
    if agent.replay_buffer.count >= 2000:
        for update in range(20):
            
            # learn
            al, cl = agent.learn()
            actor_loss += al
            critic_loss += cl
            
            # scheduler step
            agent.actor.scheduler.step()
            agent.critic.scheduler.step()
        
    # store
    returns.append(score)
    actor_errors.append(actor_loss)
    critic_errors.append(critic_loss)
    avg_return = np.mean(returns[-100:])
    avg_returns.append(avg_return)
    
    # print progress
    print('Episode no:', i, 'Score %.4f' % score, 'Average score %.4f' % avg_return)
        
# plot and save
filename = 'LunarLanderContinuous_.png'
plot_df = pd.DataFrame(0, index = np.arange(n_episodes), columns = ['x', 'Scores'])
plot_df['Episode'] = np.arange(n_episodes)
plot_df['Return'] = avg_returns
plt.figure()
sns.lineplot(data = plot_df, 
             x = 'Episode', 
             y = 'Return',
             ci = 'sd')
plt.savefig(filename)