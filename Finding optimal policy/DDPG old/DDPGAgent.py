import ACNetwork
from ReplayBuffer import ReplayBuffer
import NoiseProcess
import torch
import torch.nn.functional as F
import numpy as np
from collections import namedtuple

Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])


class Agent:
    def __init__(self, environment, lr_actor, lr_critic, 
                tau = 0.001, gamma = 0.99, noise = 'OrsteinUlhenbeck', hidden_dim1 = 64, hidden_dim2 = 32, buffer_size = 10000, batch_size = 64):

        #
        self.observation_space = environment.observation_space
        self.action_space = environment.action_space
        self.tau = tau
        self.gamma = gamma

        # Define actor network
        self.actor = ACNetwork.Actor(lr_actor, self.observation_space, hidden_dim1, hidden_dim2, self.action_space)
        self.target_actor = ACNetwork.Actor(lr_actor, self.observation_space, hidden_dim1, hidden_dim2, self.action_space)
        self.target_actor.load_state_dict(self.actor.state_dict())

        # Define critic network
        self.critic = ACNetwork.Critic(lr_critic, self.observation_space, hidden_dim1, hidden_dim2, self.action_space)
        self.target_critic = ACNetwork.Critic(lr_critic, self.observation_space, hidden_dim1, hidden_dim2, self.action_space)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Define noise process (for exploration)
        if noise == 'OrsteinUlhenbeck':
            self.noise = NoiseProcess.OrsteinUlhenbeckNoise(self.action_space)
        else:
            self.noise = NoiseProcess.NormalNoise(self.action_space)

        # replay buffer (this will be used for predicting 'target' part of the TD-error, via target networks)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.sample_batch_size = batch_size

    def save_experience(self, s, a, r, s_next, done):

        experience = Experience(s, a, r, s_next, done)
        self.replay_buffer.add_experience(experience)

        return

    def target_network_update(self):

        # update weights of the target actor network
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(((1 - self.tau) * target_param) + (self.tau * param))

        # update weights of the target critic network
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(((1 - self.tau) * target_param) + (self.tau * param))

        return

    def action_selection(self, observation):
        
        self.actor.eval()

        # use actor NN to select the action
        action = self.actor.forward(observation)
        action = action.detach().numpy()

        # add noise
        action = action + self.noise.sample()
        
        # clip action values
        action = np.clip(action, self.action_space.low, self.action_space.high)

        return action

    def TD_target(self, reward, observation_next):

        # we use the target networks to compute the target part of the TD-error
        action_next = self.target_actor.forward(observation_next)
        q_next = self.target_critic.forward(observation_next, action_next)
        td_target = reward + torch.mul(self.gamma, q_next)

        return td_target

    def learn(self):        

        # first sample a batch from the replay buffer
        s_b, a_b, r_b, s_next_b, done_b = self.replay_buffer.sample_batch()

        #
        #self.actor.eval()
        #self.critic.eval()
        #self.target_actor.eval()
        #self.target_critic.eval()
        
        # compute td target via target networls
        with torch.no_grad():
            td_target = self.TD_target(r_b, s_next_b)

        # compute TD-error
        q_value = self.critic.forward(s_b, a_b)
        critic_loss = F.smooth_l1_loss(td_target, q_value)

        # update critic network
        #self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        #self.critic.eval()

        # actor loss
        a_b_prediction = self.actor.forward(s_b)
        actor_loss = -self.critic(s_b, a_b_prediction).mean()

        # update actor network
        #self.actor.train()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        #self.actor.eval()

        # update target network
        self.target_network_update()

        return actor_loss, critic_loss






