import os
import torch
import numpy as np
from torch.nn import functional as F
from .model import CriticNet, Actor_1_Net, Actor_2_Net, Actor_3_Net
from .Image_model import CNN, Ray, Feature
from .replay_memory import Transition, Memory


class MultiDDPG_Agent:
    def __init__(self, **config):
        self.config = config
        self.n_states = self.config["n_states"]
        self.n_actions = self.config["n_actions"]
        self.batch_size = self.config["batch_size"]
        self.actor_lr = self.config["actor_lr"]
        self.critic_lr = self.config["critic_lr"]
        self.gamma = self.config["gamma"]
        self.sigma = self.config["sigma"]
        self.tau = self.config["tau"]
        self.memory = Memory(self.config["mem_size"], self.config["seed"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.critic = CriticNet(n_states=self.n_states, n_actions=self.n_actions,
                                n_hidden_filters=self.config['n_hidden_filters']).to(self.device)

        self.actor_1 = Actor_1_Net(n_actions=self.n_actions, action_bound=self.config["action_bound"]).to(self.device)
        self.actor_2 = Actor_2_Net(n_actions=self.n_actions, action_bound=self.config["action_bound"]).to(self.device)
        self.actor_3 = Actor_3_Net(n_actions=self.n_actions, action_bound=self.config["action_bound"]).to(self.device)

        self.actors_list = [self.actor_1, self.actor_2, self.actor_3]

        self.target_critic = CriticNet(n_states=self.n_states, n_actions=self.n_actions,
                                       n_hidden_filters=self.config['n_hidden_filters']).to(self.device)
        self.target_actor_1 = Actor_1_Net(n_actions=self.n_actions,
                                          action_bound=self.config["action_bound"]).to(self.device)
        self.target_actor_2 = Actor_2_Net(n_actions=self.n_actions,
                                          action_bound=self.config["action_bound"]).to(self.device)
        self.target_actor_3 = Actor_3_Net(n_actions=self.n_actions,
                                          action_bound=self.config["action_bound"]).to(self.device)

        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor_1.load_state_dict(self.actor_1.state_dict())
        self.target_actor_2.load_state_dict(self.actor_2.state_dict())
        self.target_actor_3.load_state_dict(self.actor_3.state_dict())

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.actor_1_optimizer = torch.optim.Adam(self.actor_1.parameters(), lr=self.actor_lr)
        self.actor_2_optimizer = torch.optim.Adam(self.actor_2.parameters(), lr=self.actor_lr)
        self.actor_3_optimizer = torch.optim.Adam(self.actor_3.parameters(), lr=self.actor_lr)

        if self.config["unity_camera"]:
            self.cnn = CNN().to(self.device)
            self.ray = Ray().to(self.device)
            self.feature = Feature().to(self.device)

    def take_action(self, state, actor_index):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        selected_actor = self.actors_list[actor_index]
        action = selected_actor(state)[0]
        action = action.detach().cpu().numpy()
        action = action + self.sigma * np.random.randn(self.n_actions)
        return action

    def soft_update(self, net, target_net):
        for param_target, parma in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1 - self.tau) + parma.data * self.tau)

    def update(self):
        if len(self.memory) >= self.batch_size:
            batch = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states, dones = self.unpack(batch)

            next_q1_values = self.target_critic(next_states, self.target_actor_1(next_states))
            next_q2_values = self.target_critic(next_states, self.target_actor_2(next_states))
            next_q3_values = self.target_critic(next_states, self.target_actor_3(next_states))

            q_targets_1 = rewards + self.gamma * next_q1_values * (1 - dones.int())
            q_targets_2 = rewards + self.gamma * next_q2_values * (1 - dones.int())
            q_targets_3 = rewards + self.gamma * next_q3_values * (1 - dones.int())

            critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets_1)) + \
                          torch.mean(F.mse_loss(self.critic(states, actions), q_targets_2)) + \
                          torch.mean(F.mse_loss(self.critic(states, actions), q_targets_3))

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss_1 = -torch.mean(self.critic(states, self.actor_1(states)))
            self.actor_1_optimizer.zero_grad()
            actor_loss_1.backward()
            self.actor_1_optimizer.step()
            actor_loss_2 = -torch.mean(self.critic(states, self.actor_2(states)))
            self.actor_2_optimizer.zero_grad()
            actor_loss_2.backward()
            self.actor_2_optimizer.step()
            actor_loss_3 = -torch.mean(self.critic(states, self.actor_3(states)))
            self.actor_3_optimizer.zero_grad()
            actor_loss_3.backward()
            self.actor_3_optimizer.step()

            self.soft_update(self.actor_1, self.target_actor_1)
            self.soft_update(self.actor_2, self.target_actor_2)
            self.soft_update(self.actor_3, self.target_actor_3)
            self.soft_update(self.critic, self.target_critic)

            return actor_loss_1, actor_loss_2, actor_loss_3
        else:
            return None, None, None

    def unpack(self, batch):
        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).view(self.batch_size, self.n_states).to(self.device)
        actions = torch.cat(batch.action).view(-1, self.config["n_actions"]).to(self.device)
        rewards = torch.cat(batch.reward).view(self.batch_size, 1).to(self.device)
        next_states = torch.cat(batch.next_state).view(self.batch_size, self.n_states).to(self.device)
        dones = torch.cat(batch.done).view(self.batch_size, 1).to(self.device)

        return states, actions, rewards, next_states, dones

    def store(self, state, action, reward, next_state, done):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = torch.tensor([action], dtype=torch.float).to(self.device)
        reward = torch.tensor([reward], dtype=torch.float).to(self.device)
        next_state = torch.tensor([next_state], dtype=torch.float).to(self.device)
        done = torch.BoolTensor([done]).to(self.device)

        self.memory.add(state, action, reward, next_state, done)

    def check_state(self, state_0, state_1):
        image = torch.unsqueeze(torch.tensor(state_0, dtype=torch.float), 0).to(self.device)
        ray = torch.unsqueeze(torch.tensor(state_1, dtype=torch.float), 0).to(self.device)

        image = self.cnn(image)
        ray = self.ray(ray)
        feature = self.feature(image, ray).detach().cpu().numpy().squeeze()

        return feature

    def save_checkpoints(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/UGV/01/3actors'):
            os.makedirs('checkpoints/UGV/01/3actors')
        if ckpt_path is None:
            ckpt_path = "checkpoints/UGV/01/3actors/Multi-DDPG_checkpoint_{}_{}".format(env_name, suffix)
            print(ckpt_path)
        print('Saving models to {}'.format(ckpt_path))
        if self.config["unity_camera"]:
            torch.save({'critic': self.critic.state_dict(),
                        'actor_1': self.actor_1.state_dict(),
                        'actor_2': self.actor_2.state_dict(),
                        'actor_3': self.actor_3.state_dict(),
                        'target_critic': self.target_critic.state_dict(),
                        'target_actor_1': self.target_actor_1.state_dict(),
                        'target_actor_2': self.target_actor_2.state_dict(),
                        'target_actor_3': self.target_actor_3.state_dict(),
                        'CNN': self.cnn.state_dict(),
                        'Ray': self.ray.state_dict(),
                        'Feature': self.feature.state_dict()}, ckpt_path)
        else:
            torch.save({'critic': self.critic.state_dict(),
                        'actor_1': self.actor_1.state_dict(),
                        'actor_2': self.actor_2.state_dict(),
                        'actor_3': self.actor_3.state_dict(),
                        'target_critic': self.target_critic.state_dict(),
                        'target_actor_1': self.target_actor_1.state_dict(),
                        'target_actor_2': self.target_actor_2.state_dict(),
                        'target_actor_3': self.target_actor_3.state_dict()}, ckpt_path)
