import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNet(nn.Module):
    def __init__(self, n_obs, n_states, n_hiddens):
        super(FeatureNet, self).__init__()
        self.n_obs = n_obs
        self.n_states = n_states
        self.n_hiddens = n_hiddens

        self.hidden1 = nn.Linear(in_features=self.n_obs, out_features=self.n_hiddens)
        self.hidden2 = nn.Linear(in_features=self.n_hiddens, out_features=self.n_hiddens)
        self.hidden3 = nn.Linear(in_features=self.n_hiddens, out_features=self.n_states)
