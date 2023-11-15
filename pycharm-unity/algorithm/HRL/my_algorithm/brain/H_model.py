import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)


class Discriminator(nn.Module):
    def __init__(self, n_states, n_skills, n_hidden_filters=256):
        super(Discriminator, self).__init__()
        self.n_states = n_states
        self.n_skills = n_skills
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.q = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_skills)
        init_weight(self.q, initializer="xavier uniform")

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        logits = self.q(x)
        return logits


class HighValueNet(nn.Module):
    def __init__(self, n_states, n_hidden_filters):
        super(HighValueNet, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters

        self.fc1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        self.fc2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        self.fc3 = nn.Linear(in_features=self.n_hidden_filters, out_features=1)

    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class HighPolicyNet(nn.Module):
    def __init__(self, n_states, n_skills, n_hidden_filters, if_train):
        super(HighPolicyNet, self).__init__()
        self.n_states = n_states
        self.n_skills = n_skills
        self.n_hidden_filters = n_hidden_filters
        self.if_train = if_train

        self.fc1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        self.fc2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        self.fc3 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_skills)
        self.termination_layer = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_skills)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1), self.termination_layer(x)

    def get_termination(self, state, current_z):
        termination = self(state)[1][0][current_z].sigmoid()
        if self.if_train:
            skill_termination = torch.distributions.Bernoulli(termination)
        else:
            skill_termination = (termination > 0.5)
        return bool(skill_termination)


# class TerminationNet(nn.Module):
#     def __init__(self, n_states, n_skills, n_hidden_filters, if_train):
#         super(TerminationNet, self).__init__()
#         self.n_states = n_states
#         self.n_skills = n_skills
#         self.n_hidden_filters = n_hidden_filters
#         self.if_train = if_train
#
#         self.fc1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
#         self.fc2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
#         self.fc3 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_skills)
#
#     def forward(self, state):
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)
#
#     def get_termination(self, state, current_z):
#         termination = self(state)[:, current_z].sigmoid()
#         if self.if_train:
#             skill_termination = torch.distributions.Bernoulli(termination)
#         else:
#             skill_termination = (termination > 0.5)
#         return bool(skill_termination)
