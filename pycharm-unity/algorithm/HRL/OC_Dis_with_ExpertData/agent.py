import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from net import OptionCritic
import pickle
from torch import from_numpy
from mlagents.trainers import demo_loader


class OptionCriticAgent:
    def __init__(self, **config):
        self.config = config
        self.n_states = self.config["n_states"]
        self.n_actions = self.config["n_actions"]
        self.n_options = self.config["n_options"]
        self.lr = self.config["lr"]
        self.gamma = self.config["gamma"]
        self.tau = self.config["tau"]
        self.batch_size = self.config["batch_size"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # self.demo_path = self.config["demo_path"]
        self.pkl_exist = True
        self.expert_buffer = None
        self.file = None

        # self.log_action_probs = None  # ln π(at | st)
        # self.action_porbs = None  # π(at | st)
        # self.action_value = None  # Q(st, at)
        #
        # self.log_option_probs = None  # ln π(ot | st)
        # self.option_probs = None  # π(ot | st)
        # self.option_value = None  # V(st)

        self.softmax = nn.Softmax(dim=-1).to(self.device)

        self.steps_done = 0

        self.net = OptionCritic(self.n_states, self.n_actions, self.n_options).to(self.device)
        self.target_net = OptionCritic(self.n_states, self.n_actions, self.n_options).to(self.device)

        self.optimizer = torch.optim.Adam(self.target_net.parameters(), lr=self.lr)

        self.target_net.load_state_dict(self.net.state_dict())

    def select_action(self, state):
        self.steps_done += 1
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        option_action_probs, option_values, action_values = self.net(state)
        # option_action_probs = option_action_probs.detach().numpy()
        # option_values = option_values.detach().numpy()

        # Unity
        # option_values_tensor = option_values[0][0]
        # Gym
        option_values_tensor = option_values[0]
        option_values = option_values_tensor.detach().cpu().numpy()
        option = np.random.choice(np.arange(self.n_options), p=option_values)

        p_sa_given_o = option_action_probs[0][option]
        action = np.random.choice(np.arange(self.n_actions),
                                  p=p_sa_given_o.detach().cpu().numpy())

        return option, action

    def unpack_ExpertData(self, batch_size):
        expert_batch = random.sample(self.expert_buffer, batch_size)
        expert_batch = np.stack(expert_batch)
        expert_batch = torch.FloatTensor(expert_batch).to(self.device)
        return expert_batch[:, : self.config["n_states"]], \
               expert_batch[:, self.config["n_states"]: self.config["n_states"] + 1], \
               expert_batch[:, self.config["n_states"] + 1: self.config["n_states"] + 2], \
               expert_batch[:, self.config["n_states"] + 2: self.config["n_states"] + 2 + self.config["n_states"]], \
               expert_batch[:, self.config["n_states"] + 2 + self.config["n_states"]:]

    def update(self):
        if self.pkl_exist:
            self.file = open(self.config["pkl_filePath"], 'rb')
            self.expert_buffer = pickle.load(self.file)

        expert_states, expert_actions, expert_rewards, expert_next_states, expert_dones = self.unpack_ExpertData(batch_size=self.config["batch_size"])

        # self.replay_buffer.append([state, next_state, reward, done])
        # if len(self.replay_buffer) < self.config["batch_size"]:
        #     return
        # samples = np.array(self.replay_buffer)[
        #     np.random.choice(len(self.replay_buffer), self.config["batch_size"], replace=False)]
        # state, next_state, reward, done = np.stack(samples[:, 0]), np.stack(samples[:, 1]), \
        #                                   np.stack(samples[:, 2]), np.stack(samples[:, 3])
        # state, next_state, reward, done = torch.tensor([state], dtype=torch.float, device=self.device), \
        #                                   torch.tensor([next_state], dtype=torch.float, device=self.device), \
        #                                   torch.tensor([reward], dtype=torch.float, device=self.device), \
        #                                   torch.tensor([done], dtype=torch.bool, device=self.device)



        _, next_option_value, _ = self.target_net(expert_next_states)
        target_q = expert_rewards + (1 - expert_dones) * self.gamma * next_option_value.mean(dim=-1, keepdim=True)

        _, option_value, action_value = self.net(expert_states)
        q_value_loss = (action_value - target_q.detach()).pow(2).mean()

        _, next_option_value, _ = self.target_net(expert_next_states)
        target_v = target_q.mean(dim=-1, keepdim=True)

        option_value_loss = (option_value[:, 0].unsqueeze(-1) - target_v.detach()).pow(2).mean()

        log_prob_o = torch.log(option_value)
        o_entropy = -(torch.exp(log_prob_o) * log_prob_o).sum(dim=-1, keepdim=True)
        advantage = (target_q.detach() - action_value).mean(dim=-1, keepdim=True)
        actor_loss = -(o_entropy + self.softmax(log_prob_o - advantage.detach()) * action_value).mean()

        self.optimizer.zero_grad()
        (q_value_loss + option_value_loss + actor_loss).backward()
        self.optimizer.step()

        if self.steps_done % self.config["target_update"] == 0:
            for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
                target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

        return (q_value_loss + option_value_loss + actor_loss).item()

    def save_checkpoints(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/Pyramid'):
            os.makedirs('checkpoints/Pyramid')
        if ckpt_path is None:
            ckpt_path = "checkpoints/Pyramid/OC_dis_Ex_checkpoint_{}_{}".format(env_name, suffix)
            print(ckpt_path)
        print('Saving models to {}'.format(ckpt_path))
        torch.save(self.net.state_dict(), ckpt_path)

    def load_checkpoints(self, path):
        self.net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))

        self.net.eval()
        self.target_net.eval()
