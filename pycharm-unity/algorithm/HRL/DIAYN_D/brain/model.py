from abc import ABC
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical


def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)


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


class Discriminator(nn.Module, ABC):
    def __init__(self, n_features, n_skills):
        super(Discriminator, self).__init__()
        self.n_features = n_features
        self.n_skills = n_skills

        self.q = nn.Linear(in_features=self.n_features, out_features=self.n_skills)
        init_weight(self.q, initializer="xavier uniform")
        self.q.bias.data.zero_()

    def forward(self, features):
        logits = self.q(features)
        return logits


class ValueNetwork(nn.Module, ABC):
    def __init__(self, n_features):
        super(ValueNetwork, self).__init__()
        self.n_features = n_features

        self.value = nn.Linear(in_features=self.n_features, out_features=1)
        init_weight(self.value, initializer="xavier uniform")
        self.value.bias.data.zero_()

    def forward(self, features):
        return self.value(features)

class QvalueNetwork(nn.Module, ABC):
    def __init__(self, n_features):
        super(QvalueNetwork, self).__init__()
        self.n_features = n_features

        self.q_value = nn.Linear(in_features=self.n_features + 1, out_features=1)
        init_weight(self.q_value, initializer="xavier uniform")
        self.q_value.bias.data.zero_()

    def forward(self, features, actions):
        # x = torch.cat([states, actions], dim=1)

        # miniGrid
        x = torch.cat([features, actions.unsqueeze(1)], dim=1)
        return self.q_value(x)

class PolicyNetwork(nn.Module, ABC):
    def __init__(self, n_features, n_actions):
        super(PolicyNetwork, self).__init__()
        self.n_features = n_features
        self.n_actions = n_actions

        self.mu = nn.Linear(in_features=self.n_features, out_features=self.n_actions)
        init_weight(self.mu, initializer="xavier uniform")
        self.mu.bias.data.zero_()

        self.log_std = nn.Linear(in_features=self.n_features, out_features=self.n_actions)
        init_weight(self.log_std, initializer="xavier uniform")
        self.log_std.bias.data.zero_()

    def forward(self, features):
        mu = self.mu(features)
        log_std = self.log_std(features)
        std = log_std.clamp(min=-20, max=2).exp()
        prob = F.softmax(mu, dim=-1)
        dist = Categorical(probs=prob)
        return dist

    def sample_or_likelihood(self, features):
        dist = self(features)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob
