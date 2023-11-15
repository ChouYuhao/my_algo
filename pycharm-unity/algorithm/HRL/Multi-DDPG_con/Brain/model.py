import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CriticNet(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden_filters=64):
        super(CriticNet, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_actions = n_actions

        self.fc1 = nn.Linear(in_features=self.n_states + self.n_actions, out_features=self.n_hidden_filters)
        self.fc2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        self.fc3 = nn.Linear(in_features=self.n_hidden_filters, out_features=1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# class FeatureNet(nn.Module):
#     def __init__(self, n_states, n_features=32, n_hidden_filters=256):
#         super(FeatureNet, self).__init__()
#         self.n_states = n_states
#         self.n_features = n_features
#         self.n_hidden_filters = n_hidden_filters
#
#         self.fc1 = nn.Linear(in_features=self.n_states, out_features=self.n_hiddens)
#         self.fc2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
#         self.fc3 = nn.Linear(in_features=self.n_hiddens, out_features=self.n_features)
#
#     def forward(self, state):
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         state_feature = F.relu(self.fc3(x))
#         return state_feature


class Actor_1_Net(nn.Module):
    def __init__(self, n_actions, action_bound, n_features=32, n_hidden_filters=64):
        super(Actor_1_Net, self).__init__()
        self.n_actions = n_actions
        self.action_bound = action_bound
        self.n_features = n_features
        self.n_hidden_filters = n_hidden_filters

        self.fc1 = nn.Linear(in_features=self.n_features, out_features=self.n_hidden_filters)
        self.fc2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)

    def forward(self, state_feature):
        state_feature = F.relu(self.fc1(state_feature))
        action = torch.tanh(self.fc2(state_feature))
        action = (action * self.action_bound[1]).clamp_(self.action_bound[0], self.action_bound[1])
        return action


class Actor_2_Net(nn.Module):
    def __init__(self, n_actions, action_bound, n_features=32, n_hidden_filters=64):
        super(Actor_2_Net, self).__init__()
        self.n_actions = n_actions
        self.action_bound = action_bound
        self.n_features = n_features
        self.n_hidden_filters = n_hidden_filters

        self.fc1 = nn.Linear(in_features=self.n_features, out_features=self.n_hidden_filters)
        self.fc2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)

    def forward(self, state_feature):
        state_feature = F.relu(self.fc1(state_feature))
        action = torch.tanh(self.fc2(state_feature))
        action = (action * self.action_bound[1]).clamp_(self.action_bound[0], self.action_bound[1])
        return action


class Actor_3_Net(nn.Module):
    def __init__(self, n_actions, action_bound, n_features=32, n_hidden_filters=64):
        super(Actor_3_Net, self).__init__()
        self.n_actions = n_actions
        self.action_bound = action_bound
        self.n_features = n_features
        self.n_hidden_filters = n_hidden_filters

        self.fc1 = nn.Linear(in_features=self.n_features, out_features=self.n_hidden_filters)
        self.fc2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)

    def forward(self, state_feature):
        state_feature = F.relu(self.fc1(state_feature))
        action = torch.tanh(self.fc2(state_feature))
        action = (action * self.action_bound[1]).clamp_(self.action_bound[0], self.action_bound[1])
        return action


