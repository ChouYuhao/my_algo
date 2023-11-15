import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


class LowPolicyNet(nn.Module):
    def __init__(self, n_states, n_actions, action_bound, n_hidden_filters=256):
        super(LowPolicyNet, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden_filters = n_hidden_filters
        self.action_bound = action_bound

        self.fc1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        self.fc2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        self.fc_mu = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        self.fc_std = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)

    def forward(self, state_):
        x = F.relu(self.fc1(state_))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        # rsample() 是重参数采样
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(value=normal_sample)
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的概率密度函数
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)
        action = (action * self.action_bound[1]).clamp_(self.action_bound[0], self.action_bound[1])
        return action, log_prob


class LowValueNet(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden_filters=256):
        super(LowValueNet, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden_filters = n_hidden_filters

        self.fc1 = nn.Linear(in_features=self.n_states + self.n_actions, out_features=self.n_hidden_filters)
        self.fc2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        self.fc3 = nn.Linear(in_features=self.n_hidden_filters, out_features=1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
