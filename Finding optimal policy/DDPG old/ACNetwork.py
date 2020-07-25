import NoiseProcess as Noise
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, learning_rate, observation_space, hidden_dim1, hidden_dim2, action_space):
        super(Actor, self).__init__()

        # network's structural attributes
        self.action_space = action_space
        self.observation_space = observation_space
        self.input_dim = self.observation_space.shape[0]
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = self.action_space.shape[0]

        #
        self.lr = learning_rate

        # define network
        self.layer1 = nn.Linear(self.input_dim, self.hidden_dim1)
        self.layer2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.layer3 = nn.Linear(self.hidden_dim2, self.output_dim)
        self.layer1_normalization = nn.LayerNorm(self.hidden_dim1)
        self.layer2_normalization = nn.LayerNorm(self.hidden_dim2)
        self.layer_activation = nn.ReLU()
        self.layer3_activation = nn.Tanh()

        # weight initialization
        for name, param in self.named_parameters():
            if name in ['layer3.weight', 'layer3.bias']:
                # metioned in the DDPG paper
                factor = 0.003
            else:
                factor = 1/np.sqrt(param.data.size()[0])
            nn.init.uniform_(param.data, -factor, factor)

        # define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = 300, gamma = 0.5)

        # define device and send the netowrk to device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):

        # NOTE: Difference between actor-critic (for continuous action) and DDPG is that
        # actor predicts parameters of a stochastic policy in former and actor predicts
        # directly the real value of each action feature in latter.

        # covert to torch tensor
        observation = torch.tensor(observation, dtype = torch.float).to(self.device)

        # forward pass
        """
        action = self.layer_activation(self.layer1_normalization(self.layer1(observation)))
        action = self.layer_activation(self.layer2_normalization(self.layer2(action)))
        action = self.layer3_activation(self.layer3(action))
        """
        action = self.layer_activation(self.layer1(observation))
        action = self.layer_activation(self.layer2(action))
        action = self.layer3_activation(self.layer3(action))
        

        # in actual problem, action features are continuous but only take positive value
        # hence, it makes sense to normalize the data between 0 to 1
        if False:
            action_scaled = self.action_scaling(action)

        # following step is required to scale the output of NN (-1, 1) to action scale of action features
        action_scaled = torch.mul(action, torch.from_numpy(self.action_space.high).float())
        
        return action_scaled

    def action_scaling(self, action):

        #
        FEATURES = action.shape

        # apply min-max scaler 
        #max1, max2, min1, min2 = torch.max(action[:FEATURES/2]), torch.max(action[FEATURES/2:]), torch.min(action[:FEATURES/2]), torch.min(action[FEATURES/2:])
        # because we are using tanh activation in the last layer
        min_, max_ = -1, 1

        #
        #action[:FEATURES/2] = (action[:FEATURES/2] - min1)/(max1 - min1)
        action = (action - min_)/(max_ - min_)

        return action

class Critic(nn.Module):
    def __init__(self, learning_rate, observation_space, hidden_dim1, hidden_dim2, action_space):
        super(Critic, self).__init__()

        # network's structural attributes
        self.action_space = action_space
        self.observation_space = observation_space
        self.input_dim = self.observation_space.shape[0]
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = 1

        #
        self.lr = learning_rate

        # network definition
        # NOTE: DDPG paper has a peculiar structure for critic NN, which is different from the Actor structure
        # First we pass the 'observation' through a FF network, to size of 400, activated by ReLU.
        # I imagined this output from the first pass as some feature representation of the state value function.
        # Not sure if it is correct to make this analogy.
        # Then this hidden representation is cancatenated with action vector and concatenated vector is passed
        # through another FF, to size of 300, activated by ReLU. If we want to continue the previously made
        # analogy then of course, we want to probe into the state-values to escavate to state-action-values.
        # For this purpose we concatenate the action vector to the state-value vector pass through NN. 
        # After which we get a hidden representation of the state-action value fucntion, which is then passed
        # through another FF layer, to size action_feature, not activated.
        
        """
        self.layer1 = nn.Linear(self.input_dim, self.hidden_dim1)
        self.layer2 = nn.Linear(self.hidden_dim1 + self.output_dim, self.hidden_dim2)
        self.layer3 = nn.Linear(self.hidden_dim2, 1)
        self.layer_activation = nn.ReLU()
        self.layer1_normalization = nn.LayerNorm(self.hidden_dim1)
        """
        self.s_expansion = nn.Linear(self.input_dim, self.hidden_dim1)
        self.a_expansion = nn.Linear(self.action_space.shape[0], self.hidden_dim1)
        self.ff_1 = nn.Linear(self.hidden_dim1*2, self.hidden_dim2)
        self.ff_2 = nn.Linear(self.hidden_dim2, self.output_dim)
        self.layer_activation = nn.ReLU()
        
        # weight initialization
        for name, param in self.named_parameters():
            if name in ['layer3.weight', 'layer3.bias']:
                # metioned in the DDPG paper
                factor = 0.003
            else:
                factor = 1/np.sqrt(param.data.size()[0])
            nn.init.uniform_(param.data, -factor, factor)

        # define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = 300, gamma = 0.5)

        # define device and send the netowrk to device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation, action):
        
        # convert to torch tensor
        observation, action = torch.tensor(observation, dtype = torch.float).to(self.device), torch.tensor(action, dtype = torch.float).to(self.device)

        # forward pass
        """
        x = self.layer_activation(self.layer1_normalization(self.layer1(observation)))
        y = self.layer_activation(self.layer2(torch.cat([x, action], dim = 1)))
        q_values = self.layer3(y)
        """
        s_expanded = self.layer_activation(self.s_expansion(observation))
        a_expanded = self.layer_activation(self.a_expansion(action))
        q_values = self.layer_activation(self.ff_1(torch.cat([s_expanded, a_expanded], dim = 1)))
        q_values = self.ff_2(q_values)

        return q_values





