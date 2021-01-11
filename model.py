import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


class Actor(nn.Module):
    """ Policy Model """

    def __init__(self, state_size, action_size, seed):
        
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.reset_parameters()
        
    def hidden_init(self, layer):
        
        hid_layer = layer.weight.data.size()[0]
        lim = 1.0 / np.sqrt(hid_layer)
        return (-lim, lim)
        
    def reset_parameters(self):
        
        self.fc1.weight.data.uniform_(*self.hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*self.hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)        
        x = self.fc3(x)
        x = F.tanh(x)
        return x


class Critic(nn.Module):
    """ Value Model """

    def __init__(self, state_size, action_size, seed):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128+action_size, 128)
        self.fc3 = nn.Linear(128, 1)
        self.reset_parameters()
        
    def hidden_init(self, layer):
        hid_layer = layer.weight.data.size()[0]
        lim = 1.0 / np.sqrt(hid_layer)
        return (-lim, lim)
        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*self.hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*self.hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = torch.cat((x, action), dim=1)
        x = self.fc2(x)
        x = F.relu(x)        
        x = self.fc3(x)
        return x