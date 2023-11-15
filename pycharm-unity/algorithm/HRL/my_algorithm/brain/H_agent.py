import os
import torch
import numpy as np
from .H_model import Discriminator, HighValueNet, HighPolicyNet
from .L_model import LowValueNet, LowPolicyNet
from torch import nn
from torch.nn import functional as F
from .replay_memory import Memory, Transition
from .Image_model import CNN, Ray, Feature


# 上层网络的输入(状态)为state, 下层网络的输入(状态)为cat(state, skill)
class HRL_Agent:
    def __init__(self, p_z, **config):
        self.config = config
        self.n_states = self.config["n_states"]
        self.n_skills = self.config["n_skills"]
        self.n_actions = self.config["n_actions"]
        self.n_hidden_filters = self.config["n_hidden_filters"]
        self.H_batch_size = self.config["H_batch_size"]
        # self.H_actor_lr = self.config["H_actor_lr"]
        # self.H_critic_lr = self.config["H_critic_lr"]
        # self.H_alpha_lr = self.config["H_alpha_lr"]
        self.p_z = np.tile(p_z, self.H_batch_size).reshape(self.H_batch_size, self.n_skills)
        self.lr = self.config["lr"]
        self.actor_lr = self.config["actor_lr"]
        self.critic_lr = self.config["critic_lr"]
        self.gamma = self.config["gamma"]
        self.target_entropy = self.config["target_entropy"]
        self.tau = self.config["tau"]
        self.H_memory = Memory(self.config["H_mem_size"], self.config["seed"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 高层agent
        self.H_actor = HighPolicyNet(self.n_states, self.n_skills, self.n_hidden_filters,
                                     self.config["do_train"]).to(self.device)
        self.H_critic_1 = HighValueNet(self.n_states, self.n_hidden_filters).to(self.device)
        self.H_critic_2 = HighValueNet(self.n_states, self.n_hidden_filters).to(self.device)
        self.H_target_critic_1 = HighValueNet(self.n_states, self.n_hidden_filters).to(self.device)
        self.H_target_critic_2 = HighValueNet(self.n_states, self.n_hidden_filters).to(self.device)
        self.discriminator = Discriminator(self.n_states, self.n_skills, self.n_hidden_filters).to(self.device)
        # self.termination = TerminationNet(self.n_states, self.n_skills, self.n_hidden_filters,
        #                                   self.config["do_train"]).to(self.device)

        self.H_target_critic_1.load_state_dict(self.H_critic_1.state_dict())
        self.H_target_critic_2.load_state_dict(self.H_critic_2.state_dict())

        self.H_actor_optimizer = torch.optim.Adam(self.H_actor.parameters(), lr=self.actor_lr)
        self.H_critic_1_optimizer = torch.optim.Adam(self.H_critic_1.parameters(), lr=self.critic_lr)
        self.H_critic_2_optimizer = torch.optim.Adam(self.H_critic_2.parameters(), lr=self.critic_lr)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)

        # self.termination_optimizer = torch.optim.Adam(self.termination.parameters(), lr=self.lr)

        # 底层agent
        self.L_actor = LowPolicyNet(self.n_states + self.n_skills, self.n_actions, self.config["action_bound"],
                                    self.n_hidden_filters).to(self.device)
        self.L_critic_1 = LowValueNet(self.n_states + self.n_skills, self.n_actions,
                                      self.n_hidden_filters).to(self.device)
        self.L_critic_2 = LowValueNet(self.n_states + self.n_skills, self.n_actions,
                                      self.n_hidden_filters).to(self.device)
        self.L_target_critic_1 = LowValueNet(self.n_states + self.n_skills, self.n_actions,
                                             self.n_hidden_filters).to(self.device)
        self.L_target_critic_2 = LowValueNet(self.n_states + self.n_skills, self.n_actions,
                                             self.n_hidden_filters).to(self.device)

        self.L_target_critic_1.load_state_dict(self.L_critic_1.state_dict())
        self.L_target_critic_2.load_state_dict(self.L_critic_2.state_dict())

        self.L_actor_optimizer = torch.optim.Adam(self.L_actor.parameters(), lr=self.actor_lr)
        self.L_critic_1_optimizer = torch.optim.Adam(self.L_critic_1.parameters(), lr=self.critic_lr)
        self.L_critic_2_optimizer = torch.optim.Adam(self.L_critic_2.parameters(), lr=self.critic_lr)

        # 使用alpha的log值,可以使训练结果比较稳定
        self.H_log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.H_log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.H_log_alpha_optimizer = torch.optim.Adam([self.H_log_alpha], lr=self.critic_lr)

        self.L_log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.L_log_alpha.requires_grad = True
        self.L_log_alpha_optimizer = torch.optim.Adam([self.L_log_alpha], lr=self.critic_lr)

        if self.config["unity_camera"]:
            self.cnn = CNN().to(self.device)
            self.ray = Ray().to(self.device)
            self.feature = Feature().to(self.device)

    # 高层选择策略 state 为 s
    def take_skill(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.H_actor(state)[0]
        skill_dist = torch.distributions.Categorical(probs)
        skill = skill_dist.sample()
        return skill.item()

    def get_termination(self, s, current_z):
        s = torch.tensor([s], dtype=torch.float).to(self.device)
        return self.H_actor.get_termination(s, current_z)

    # 下层选择动作 state 为 s + z
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.L_actor(state)[0]
        return action.detach().cpu().numpy()[0]

    # 计算高层td_target
    def calc_skill_target(self, next_states, zs, dones):
        next_probs = self.H_actor(torch.split(next_states, [self.n_states, self.n_skills], dim=-1)[0])[0]
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.H_target_critic_1(torch.split(next_states, [self.n_states, self.n_skills], dim=-1)[0])
        q2_value = self.H_target_critic_2(torch.split(next_states, [self.n_states, self.n_skills], dim=-1)[0])
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        next_value = min_qvalue + self.H_log_alpha.exp() * entropy

        logits = self.discriminator(torch.split(next_states, [self.n_states, self.n_skills], dim=-1)[0])
        p_z = torch.tensor(self.p_z, dtype=torch.float).to(self.device)
        p_z = p_z.gather(-1, zs)
        logq_z_ns = F.log_softmax(logits, dim=-1)
        rewards = logq_z_ns.gather(-1, zs).detach() - torch.log(p_z + 1e-6)

        H_td_target = rewards + self.gamma * next_value * (1 - dones.int())
        return H_td_target

    # 计算底层td_target
    def calc_action_target(self, rewards, next_states, dones):
        next_actions, log_prob = self.L_actor(next_states)
        entropy = -log_prob
        q1_value = self.L_target_critic_1(next_states, next_actions)
        q2_value = self.L_target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) + self.L_log_alpha.exp() * entropy
        L_td_target = rewards + self.gamma * next_value + (1 - dones.int())
        return L_td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1 - self.tau) + param.data * self.tau)

    def update(self):
        count = 0
        H_actor_loss = 0
        L_actor_loss = 0
        if len(self.H_memory) < self.H_batch_size:
            return None, None
        else:
            batch = self.H_memory.sample(self.H_batch_size)
            # state = s + z
            states, zs, actions, rewards, next_states, dones = self.unpack(batch)

            rewards = (rewards + 10.0) / 10.0

            if count % self.config['H'] == 0:
                # 更新高层Q网络
                with torch.no_grad():
                    H_td_target = self.calc_skill_target(next_states, zs, dones)
                H_critic_1_q_values = self.H_critic_1(torch.split(next_states, [self.n_states, self.n_skills], dim=-1)[0])
                H_critic_1_loss = torch.mean(F.mse_loss(H_critic_1_q_values, H_td_target.detach()))
                H_critic_2_q_values = self.H_critic_2(torch.split(next_states, [self.n_states, self.n_skills], dim=-1)[0])
                H_critic_2_loss = torch.mean(F.mse_loss(H_critic_2_q_values, H_td_target.detach()))
                self.H_critic_1_optimizer.zero_grad()
                H_critic_1_loss.backward()
                self.H_critic_1_optimizer.step()
                self.H_critic_2_optimizer.zero_grad()
                H_critic_2_loss.backward()
                self.H_critic_2_optimizer.step()

                # 更新上层策略网络
                probs = self.H_actor(torch.split(next_states, [self.n_states, self.n_skills], dim=1)[0])[0]
                log_probs = torch.log(probs + 1e-8)
                H_entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
                q1_value = self.H_critic_1(torch.split(next_states, [self.n_states, self.n_skills], dim=1)[0])
                q2_value = self.H_critic_2(torch.split(next_states, [self.n_states, self.n_skills], dim=1)[0])
                min_qvalue = torch.sum(probs * torch.min(q2_value, q1_value), dim=1, keepdim=True)
                H_actor_loss = torch.mean(-self.H_log_alpha.exp() * H_entropy - min_qvalue)
                self.H_actor_optimizer.zero_grad()
                H_actor_loss.backward()
                self.H_actor_optimizer.step()

                # 更新alpha的值
                H_alpha_loss = torch.mean((H_entropy - self.target_entropy).detach() * self.H_log_alpha.exp())
                self.H_log_alpha_optimizer.zero_grad()
                H_alpha_loss.backward()
                self.H_log_alpha_optimizer.step()

                self.soft_update(self.H_critic_1, self.H_target_critic_1)
                self.soft_update(self.H_critic_2, self.H_target_critic_2)

                # 更新判别器
                logits = self.discriminator(torch.split(next_states, [self.n_states, self.n_skills], dim=-1)[0])
                discriminator_loss = F.cross_entropy(logits, zs.squeeze(-1))
                self.discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                self.discriminator_optimizer.step()

            # 更新下层Q网络
            with torch.no_grad():
                L_td_target = self.calc_action_target(rewards, next_states, dones)
            L_critic_1_loss = torch.mean(F.mse_loss(self.L_critic_1(states, actions), L_td_target.detach()))
            L_critic_2_loss = torch.mean(F.mse_loss(self.L_critic_2(states, actions), L_td_target.detach()))
            self.L_critic_1_optimizer.zero_grad()
            L_critic_1_loss.backward()
            self.L_critic_1_optimizer.step()
            self.L_critic_2_optimizer.zero_grad()
            L_critic_2_loss.backward()
            self.L_critic_2_optimizer.step()

            # 更新策略网络
            new_actions, log_prob = self.L_actor(states)
            L_entropy = -log_prob
            q1_value = self.L_critic_1(states, new_actions)
            q2_value = self.L_critic_2(states, new_actions)
            L_actor_loss = torch.mean(-self.L_log_alpha.exp() * L_entropy - torch.min(q1_value, q2_value))

            self.L_actor_optimizer.zero_grad()
            L_actor_loss.backward()
            self.L_actor_optimizer.step()

            # 更新alpha值
            L_alpha_loss = torch.mean(
                (L_entropy - self.target_entropy).detach() * self.L_log_alpha.exp())
            self.L_log_alpha_optimizer.zero_grad()
            L_alpha_loss.backward()
            self.L_log_alpha_optimizer.step()

            self.soft_update(self.L_critic_1, self.L_target_critic_1)
            self.soft_update(self.L_critic_2, self.L_target_critic_2)

            count += 1

            return H_actor_loss.item(), L_actor_loss.item()

    def store(self, state, z, action, reward, next_state, done):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        z = torch.ByteTensor([z]).to(self.device)
        action = torch.tensor([action]).to(self.device)
        reward = torch.tensor([reward], dtype=torch.float).to(self.device)
        next_state = torch.tensor([next_state], dtype=torch.float).to(self.device)
        done = torch.BoolTensor([done]).to(self.device)
        self.H_memory.add(state, z, action, reward, next_state, done)

    def unpack(self, batch):
        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).view(self.H_batch_size, self.n_states + self.n_skills).to(self.device)
        zs = torch.cat(batch.z).view(self.H_batch_size, 1).long().to(self.device)
        actions = torch.cat(batch.action).view(-1, self.config["n_actions"]).to(self.device)
        rewards = torch.cat(batch.reward).view(self.H_batch_size, 1).to(self.device)
        next_states = torch.cat(batch.next_state).view(self.H_batch_size, self.n_states + self.n_skills).to(self.device)
        dones = torch.cat(batch.done).view(self.H_batch_size, 1).to(self.device)

        return states, zs, actions, rewards, next_states, dones

    def check_state(self, state_0, state_1):
        image = torch.unsqueeze(torch.tensor(state_0, dtype=torch.float), 0).to(self.device)
        ray = torch.unsqueeze(torch.tensor(state_1, dtype=torch.float), 0).to(self.device)

        # image = self.cnn(image).detach().cpu().numpy().squeeze()
        # ray = self.ray(ray).detach().cpu().numpy().squeeze()
        image = self.cnn(image)
        ray = self.ray(ray)
        feature = self.feature(image, ray).detach().cpu().numpy().squeeze()

        # state = np.concatenate((image, ray), axis=0)
        return feature

    def save_checkpoints(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/AutoRooms/001_10skills'):
            os.makedirs('checkpoints/AutoRooms/001_10skills')
        if ckpt_path is None:
            ckpt_path = "checkpoints/AutoRooms/001_10skills/MyAlgorithm_checkpoint_{}_{}".format(env_name, suffix)
            print(ckpt_path)
        print('Saving models to {}'.format(ckpt_path))
        if self.config["unity_camera"]:
            torch.save({'H_actor': self.H_actor.state_dict(),
                        'H_critic_1': self.H_critic_1.state_dict(),
                        'H_critic_2': self.H_critic_2.state_dict(),
                        'H_target_critic_1': self.H_target_critic_1.state_dict(),
                        'H_target_critic_2': self.H_target_critic_2.state_dict(),
                        'discriminator': self.discriminator.state_dict(),
                        'H_log_alpha_optimizer': self.H_log_alpha_optimizer.state_dict(),
                        'L_actor': self.L_actor.state_dict(),
                        'L_critic_1': self.L_critic_1.state_dict(),
                        'L_critic_2': self.L_critic_2.state_dict(),
                        'L_target_critic_1': self.L_target_critic_1.state_dict(),
                        'L_target_critic_2': self.L_target_critic_2.state_dict(),
                        'L_log_alpha_optimizer': self.L_log_alpha_optimizer.state_dict(),
                        'CNN': self.cnn.state_dict(),
                        'Ray': self.ray.state_dict(),
                        'Feature': self.feature.state_dict()}, ckpt_path)
        else:
            torch.save({'H_actor': self.H_actor.state_dict(),
                        'H_critic_1': self.H_critic_1.state_dict(),
                        'H_critic_2': self.H_critic_2.state_dict(),
                        'H_target_critic_1': self.H_target_critic_1.state_dict(),
                        'H_target_critic_2': self.H_target_critic_2.state_dict(),
                        'discriminator': self.discriminator.state_dict(),
                        'H_log_alpha_optimizer': self.H_log_alpha_optimizer.state_dict(),
                        'L_actor': self.L_actor.state_dict(),
                        'L_critic_1': self.L_critic_1.state_dict(),
                        'L_critic_2': self.L_critic_2.state_dict(),
                        'L_target_critic_1': self.L_target_critic_1.state_dict(),
                        'L_target_critic_2': self.L_target_critic_2.state_dict(),
                        'L_log_alpha_optimizer': self.L_log_alpha_optimizer.state_dict()}, ckpt_path)

    def load_checkpoints(self, path):
        checkpoints = torch.load(path)
        self.H_actor.load_state_dict(checkpoints['H_actor'])
        self.H_critic_1.load_state_dict(checkpoints['H_critic_1'])
        self.H_critic_2.load_state_dict(checkpoints['H_critic_2'])
        self.H_target_critic_1.load_state_dict(checkpoints['H_target_critic_1'])
        self.H_target_critic_2.load_state_dict(checkpoints['H_target_critic_2'])
        self.discriminator.load_state_dict(checkpoints['discriminator'])
        self.H_log_alpha_optimizer.load_state_dict(checkpoints['H_log_alpha_optimizer'])
        self.L_actor.load_state_dict(checkpoints['L_actor'])
        self.L_critic_1.load_state_dict(checkpoints['L_critic_1'])
        self.L_critic_2.load_state_dict(checkpoints['L_critic_2'])
        self.L_target_critic_1.load_state_dict(checkpoints['L_target_critic_1'])
        self.L_target_critic_2.load_state_dict(checkpoints['L_target_critic_2'])
        self.L_log_alpha_optimizer.load_state_dict(checkpoints['L_log_alpha_optimizer'])

        if self.config["unity_camera"]:
            self.cnn.load_state_dict(checkpoints['CNN'])
            self.ray.load_state_dict(checkpoints['Ray'])
            self.feature.load_state_dict(checkpoints['Feature'])

            self.cnn.eval()
            self.ray.eval()
            self.feature.eval()

        self.H_actor.eval()
        self.H_critic_1.eval()
        self.H_critic_2.eval()
        self.H_target_critic_1.eval()
        self.H_target_critic_2.eval()
        self.discriminator.eval()
        self.L_actor.eval()
        self.L_critic_1.eval()
        self.L_critic_2.eval()
        self.L_target_critic_1.eval()
        self.L_target_critic_2.eval()

