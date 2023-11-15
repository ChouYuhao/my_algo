import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNFeatureNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_dim):
        super(CNNFeatureNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels[0], kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(hidden_channels[1], hidden_channels[2], kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(hidden_channels[2] * 5 * 5, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(-1, self.fc.in_features)
        x = self.fc(x)
        return self.relu(x)

    def get_states(self, obs):
        return self(obs).detach()


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
