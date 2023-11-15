import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from net import PolicyNet, QValueNet, CNNFeatureNet
from replay_memory import Transition, Memory


class SAC_d:
    def __init__(self, **config):
        self.config = config
        self.n_states = self.config["n_states"]
        self.n_hiddens = self.config["n_hiddens"]
        self.n_actions = self.config["n_actions"]
        self.actor_lr = self.config["actor_lr"]
        self.critic_lr = self.config["critic_lr"]
        self.alpha_lr = self.config["alpha_lr"]
        self.target_entropy = self.config["target_entropy"]
        self.tau = self.config["tau"]
        self.gamma = self.config["gamma"]
        self.batch_size = self.config["batch_size"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.memory = Memory(self.config["mem_size"], self.config["seed"])

        self.conv = self.config["if_conv"]
        self.hidden_channels = self.config["hidden_channels"]
        self.n_features = self.config["n_features"]

        if self.conv:
            self.cnnFeatureNet = CNNFeatureNet(in_channels=3, hidden_channels=self.hidden_channels,
                                               out_dim=self.n_features).to(self.device)
            self.n_states = self.n_features

        self.actor = PolicyNet(self.n_states, self.n_hiddens, self.n_actions).to(self.device)
        self.critic_1 = QValueNet(self.n_states, self.n_hiddens, self.n_actions).to(self.device)
        self.critic_2 = QValueNet(self.n_states, self.n_hiddens, self.n_actions).to(self.device)
        self.target_critic_1 = QValueNet(self.n_states, self.n_hiddens, self.n_actions).to(self.device)
        self.target_critic_2 = QValueNet(self.n_states, self.n_hiddens, self.n_actions).to(self.device)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=self.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=self.critic_lr)

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

    def select_action(self, state):
        if not self.conv:
            # state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            state = torch.tensor(np.array([state['image'].reshape(75)]), dtype=torch.float).to(self.device)

        if self.conv:
            state = torch.tensor(np.array(state['image']), dtype=torch.float).to(self.device)
            state = torch.transpose(state, 0, 2)
            state = torch.unsqueeze(state, dim=0)
            state = self.cnnFeatureNet.get_states(state)

        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        # td_target = rewards + self.gamma * next_value * (1 - dones)
        td_target = rewards + self.gamma * next_value * (~dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def store(self, state, action, reward, next_state, done):
        if not self.conv:
            # state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            # next_state = torch.tensor(np.array([next_state]), dtype=torch.float).to(self.device)
            state = torch.tensor(np.array([state['image'].reshape(75)]), dtype=torch.float).to(self.device)
            next_state = torch.tensor(np.array([next_state['image'].reshape(75)]), dtype=torch.float).to(self.device)

        else:
            state = torch.tensor(state['image'], dtype=torch.float).to(self.device)
            next_state = torch.tensor(next_state['image'], dtype=torch.float).to(self.device)

        action = torch.tensor([action]).to(self.device)

        reward = torch.tensor([reward], dtype=torch.float).to(self.device)
        done = torch.BoolTensor([done]).to(self.device)

        self.memory.add(state, action, reward, next_state, done)

    def unpack(self, batch):
        batch = Transition(*zip(*batch))

        if self.conv:
            states = torch.cat(batch.state).view(self.batch_size, 3, 40, 40).to(self.device)
            next_states = torch.cat(batch.next_state).view(self.batch_size, 3, 40, 40).to(self.device)
        else:
            states = torch.cat(batch.state).view(self.batch_size, self.n_states).to(self.device)
            next_states = torch.cat(batch.next_state).view(self.batch_size, self.n_states).to(self.device)

        actions = torch.cat(batch.action).view(self.batch_size, 1).to(self.device)
        rewards = torch.cat(batch.reward).view(self.batch_size, 1).to(self.device)
        dones = torch.cat(batch.done).view(self.batch_size, 1).to(self.device)

        return states, actions, rewards, next_states, dones

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = self.unpack(batch)

        if self.conv:
            states = self.cnnFeatureNet.get_states(states)
            next_states = self.cnnFeatureNet.get_states(next_states)

        td_target = self.calc_target(rewards, next_states, dones)

        critic_1_qvalues = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(F.mse_loss(critic_1_qvalues, td_target.detach()))

        critic_2_qvalues = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(F.mse_loss(critic_2_qvalues, td_target.detach()))

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)

        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)  # 直接根据概率计算期望

        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

    def save_checkpoints(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/empty'):
            os.makedirs('checkpoints/empty')
        if ckpt_path is None:
            ckpt_path = "checkpoints/empty/SAC_checkpoint_{}_{}".format(env_name, suffix)
            print(ckpt_path)
        print('Saving models to {}'.format(ckpt_path))
        if not self.conv:
            torch.save({'actor': self.actor.state_dict(),
                        'critic_1': self.critic_1.state_dict(),
                        'critic_2': self.critic_2.state_dict(),
                        'target_critic_1': self.target_critic_1.state_dict(),
                        'target_critic_2': self.target_critic_2.state_dict(),
                        'cnnFeatureNet': self.cnnFeatureNet}, ckpt_path)
        else:
            torch.save({'actor': self.actor.state_dict(),
                        'critic_1': self.critic_1.state_dict(),
                        'critic_2': self.critic_2.state_dict(),
                        'target_critic_1': self.target_critic_1.state_dict(),
                        'target_critic_2': self.target_critic_2.state_dict()}, ckpt_path)

    def load_checkpoints(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(torch.load(checkpoint['actor']))
        self.critic_1.load_state_dict(torch.load(checkpoint['critic_1']))
        self.critic_2.load_state_dict(torch.load(checkpoint['critic_2']))
        self.target_critic_1.load_state_dict(torch.load(checkpoint['target_critic_1']))
        self.target_critic_2.load_state_dict(torch.load(checkpoint['target_critic_2']))
        if self.conv:
            self.cnnFeatureNet.load_state_dict(torch.load(checkpoint['cnnFeatureNet']))

        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
        self.target_critic_1.eval()
        self.target_critic_2.eval()
        if self.conv:
            self.cnnFeatureNet.eval()
