import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from net import OptionCritic
from replay_memory import Transition, Memory


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
        self.memory = Memory(self.config["mem_size"], self.config["seed"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # self.log_action_probs = None  # ln π(at | st)
        # self.action_porbs = None  # π(at | st)
        # self.action_value = None  # Q(st, at)
        #
        # self.log_option_probs = None  # ln π(ot | st)
        # self.option_probs = None  # π(ot | st)
        # self.option_value = None  # V(st)

        self.softmax = nn.Softmax(dim=-1).to(self.device)

        self.replay_buffer = []
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
        option_values_tensor = option_values[0][0]
        # Gym
        # option_values_tensor = option_values[0]
        option_values = option_values_tensor.detach().cpu().numpy()
        option = np.random.choice(np.arange(self.n_options), p=option_values)

        p_sa_given_o = option_action_probs[0][option]
        action = np.random.choice(np.arange(self.n_actions),
                                  p=p_sa_given_o.detach().cpu().numpy())

        return option, action

    def store(self, state, action, next_state, reward, done):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = torch.tensor([action]).view(-1, 1).to(self.device)
        next_state = torch.tensor([next_state], dtype=torch.float).to(self.device)
        reward = torch.tensor([reward], dtype=torch.float).to(self.device)
        done = torch.BoolTensor([done]).to(self.device)

        self.memory.add(state, action, next_state, reward, done)

    def unpack(self, batch):
        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).view(self.batch_size, self.n_states).to(self.device)
        actions = torch.cat(batch.action).view(self.batch_size, 1).to(self.device)
        rewards = torch.cat(batch.reward).view(self.batch_size, 1).to(self.device)
        next_states = torch.cat(batch.next_state).view(self.batch_size, self.n_states).to(self.device)
        dones = torch.cat(batch.done).view(self.batch_size, 1).to(self.device)

        return states, actions, rewards, next_states, dones

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = self.unpack(batch)

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

        _, next_option_value, _ = self.target_net(next_states)
        target_q = rewards + (~dones) * self.gamma * next_option_value.mean(dim=-1, keepdim=True)

        _, option_value, action_value = self.net(states)
        q_value_loss = (action_value - target_q.detach()).pow(2).mean()

        _, next_option_value, _ = self.target_net(next_states)
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

    def save_checkpoints(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/CartPole'):
            os.makedirs('checkpoints/CartPole')
        if ckpt_path is None:
            ckpt_path = "checkpoints/CartPole/OC_dis_checkpoint_{}_{}".format(env_name, suffix)
            print(ckpt_path)
        print('Saving models to {}'.format(ckpt_path))
        torch.save(self.net.state_dict(), ckpt_path)

    def load_checkpoints(self, path):
        self.net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))

        self.net.eval()
        self.target_net.eval()
