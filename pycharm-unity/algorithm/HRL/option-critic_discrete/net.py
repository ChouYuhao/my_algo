import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class OptionCritic(nn.Module):
    def __init__(self, n_states, n_actions, n_options):
        super(OptionCritic, self).__init__()

        self.n_states = n_states
        self.n_actions = n_actions
        self.n_options = n_options

        self.fc1 = nn.Linear(n_states, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, self.n_actions)
        self.fc4 = nn.Linear(64, self.n_options)
        self.fc5 = nn.Linear(64, self.n_options * self.n_actions)

    def forward(self, state):
        feature = torch.relu(self.fc1(state))
        feature = torch.relu(self.fc2(feature))

        # 动作价值函数，某一状态下的执行每一个动作的价值
        actions_values = F.softmax(self.fc3(feature), dim=-1)
        # 技能值函数，某一状态下执行每一个技能的价值
        option_values = F.softmax(self.fc4(feature), dim=-1)
        # 技能动作值函数，某一个技能下执行每一个动作的价值
        option_action_probs = F.softmax(self.fc5(feature).view(-1, self.n_options, self.n_actions), dim=-1)

        return option_action_probs, option_values, actions_values


