import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


class PolicyNet(nn.Module):
    def __init__(self, n_states, n_hidden_filters, n_skills):
        super(PolicyNet, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_skills = n_skills

        self.fc1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        self.fc2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        self.fc3 = nn.Linear(in_features=n_hidden_filters, out_features=self.n_skills)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.softmax(x, dim=-1)
