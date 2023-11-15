from abc import ABC
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)


# class PolicyNetwork(nn.Module, ABC):
#     def __init__(self, n_states, n_actions, action_bounds, n_hidden_filters=256):
#         super(PolicyNetwork, self).__init__()
#         self.n_states = n_states
#         self.n_hidden_filters = n_hidden_filters
#         self.n_actions = n_actions
#         self.action_bounds = action_bounds
#
#         self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
#         init_weight(self.hidden1)
#         self.hidden1.bias.data.zero_()
#         self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
#         init_weight(self.hidden2)
#         self.hidden2.bias.data.zero_()
#
#         self.mu = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
#         init_weight(self.mu, initializer="xavier uniform")
#         self.mu.bias.data.zero_()
#
#         self.log_std = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
#         init_weight(self.log_std, initializer="xavier uniform")
#         self.log_std.bias.data.zero_()
#
#     def forward(self, states):
#         x = F.relu(self.hidden1(states))
#         x = F.relu(self.hidden2(x))
#
#         mu = self.mu(x)
#         log_std = self.log_std(x)
#         std = log_std.clamp(min=-20, max=2).exp()
#         dist = Normal(mu, std)
#         return dist
#
#     def sample_act(self, states):
#         dist = self(states)
#         u = dist.rsample()
#         action = torch.tanh(u)
#         return (action * self.action_bounds[1]).clamp_(self.action_bounds[0], self.action_bounds[1])

class PolicyNetwork(nn.Module):
    def __init__(self, n_states, n_actions, action_bounds, n_hidden_filters=256):
        super(PolicyNetwork, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.action_bounds = action_bounds
        self.n_hidden_filters = n_hidden_filters
        self.fc1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        self.fc2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        self.fc3 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)

    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return (action * self.action_bounds[1]).clamp_(self.action_bounds[0], self.action_bounds[1])


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 1), stride=(1, 1))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), stride=(1, 1))
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(7056, 512)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.relu3(self.conv3(x))
        x = x.reshape(1, -1)
        x = self.fc(x)
        return x


class Ray(nn.Module):
    def __init__(self):
        super(Ray, self).__init__()
        self.fc1 = nn.Linear(802, 512)
        self.fc2 = nn.Linear(512, 512)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x = self.tanh(self.fc2(self.fc1(x)))
        x = self.tanh(self.fc2(self.fc1(x)))
        return self.tanh(x)
