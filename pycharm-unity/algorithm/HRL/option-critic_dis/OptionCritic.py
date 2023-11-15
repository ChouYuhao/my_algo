import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from net import opt_cri_arch
from replay_buffer import replay_buffer
import math
from torch.utils.tensorboard import SummaryWriter
from config import get_params


class option_critic(object):
    def __init__(self, env, **config):
        self.config = config
        self.env = env
        self.max_episodes = self.config["max_episodes"]
        self.exploration = self.config["exploration"]
        self.update_freq = self.config["update_freq"]
        self.freeze_interval = self.config["freeze_interval"]
        self.batch_size = self.config["batch_size"]
        self.mem_size = self.config["mem_size"]
        self.lr = self.config["lr"]
        self.gamma = self.config["gamma"]
        self.sigma = self.config["sigma"]
        self.termination_reg = self.config["termination_reg"]
        self.epsilon_init = self.config["epsilon_init"]
        self.decay = self.config["decay"]
        self.epsilon_min = self.config["epsilon_min"]
        self.entropy_weight = self.config["entropy_weight"]
        # self.conv = conv
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.render = render
        self.conv = self.config["conv"]
        # self.if_unity = if_unity

        self.n_observations = self.config["n_observations"]
        self.n_actions = self.config["n_actions"]
        self.n_options = self.config["n_options"]

        self.epsilon = lambda x: self.epsilon_min + (self.epsilon_init - self.epsilon_min) * math.exp(
            - x / self.decay)

        self.net = opt_cri_arch(self.n_observations, self.n_actions, self.n_options, self.sigma, self.conv).to(self.device)
        self.prime_net = opt_cri_arch(self.n_observations, self.n_actions, self.n_options, self.sigma, self.conv).to(
            self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.buffer = replay_buffer(self.mem_size)
        self.count = 0
        self.weight_reward = None

    def compute_critic_loss(self):

        observations, options, rewards, next_observations, dones = self.buffer.sample(self.batch_size)

        # torch.Size([1, 256, 1, 172]) torch.Size([1, 256]) torch.Size([1, 256]) torch.Size([1, 256])

        # observations = torch.FloatTensor(observations)
        # options = torch.LongTensor(options)
        # rewards = torch.FloatTensor(rewards).unsqueeze(1)
        # next_observations = torch.FloatTensor(next_observations)
        # dones = torch.FloatTensor(dones).unsqueeze(1)
        #
        # if self.if_camera:
        #     states = self.net.get_state_(observations)
        # else:
        #     states = self.net.get_state(observations)

        states = self.net.get_state(observations)
        q_values = self.net.get_q_value(states)

        prime_next_states = self.prime_net.get_state(next_observations)
        prime_next_q_values = self.prime_net.get_q_value(prime_next_states)

        # if self.if_camera:
        #     prime_next_states = self.prime_net.get_state_(next_observations)
        # else:
        #     prime_next_states = self.prime_net.get_state(next_observations)
        #
        # prime_next_q_values = self.prime_net.get_q_value(prime_next_states)

        next_states = self.net.get_state(next_observations)
        next_betas = self.net.get_termination(next_states)
        next_beta = next_betas.gather(2, options.unsqueeze(2))  # unsqueeze(1)在第1维处增加一个维度 (2,3) => (2,1,3)
        # gather(dim, index) 按照index所给的坐标选择元素

        target_q_omega = rewards.unsqueeze(2) + (self.gamma * (~dones)).unsqueeze(2) * (
                (1 - next_beta) * prime_next_q_values.gather(2, options.unsqueeze(2)) + next_beta *
                prime_next_q_values.max(1)[0].unsqueeze(1))
        td_error = (target_q_omega.detach() - q_values.gather(2, options.unsqueeze(2))).pow(2).mean()
        return td_error

    def compute_actor_loss(self, obs, option, log_prob, entropy, reward, done, next_obs, ):
        # obs = torch.FloatTensor(np.expand_dims(obs, 0))
        # next_obs = torch.FloatTensor(np.expand_dims(next_obs, 0))

        state = self.net.get_state(obs)
        next_state = self.net.get_state(next_obs)
        prime_next_state = self.prime_net.get_state(next_obs)

        next_beta = self.net.get_termination(next_state)[:, option]
        beta = self.net.get_termination(state)[:, option]

        q_value = self.net.get_q_value(state)
        next_q_value = self.net.get_q_value(next_state)
        prime_next_q_value = self.prime_net.get_q_value(next_state)

        gt = reward + self.gamma * (1 - done) * (
                (1 - next_beta) * prime_next_q_value[:, option] + next_beta * prime_next_q_value.max().unsqueeze(0))

        termination_loss = next_beta * (
                (next_q_value[:, option] - next_q_value.max().unsqueeze(0)).detach() + self.termination_reg) * (
                                   1 - done)

        policy_loss = -log_prob * (gt - q_value[:, option]).detach() - self.entropy_weight * entropy
        actor_loss = termination_loss + policy_loss
        return actor_loss

    def save_checkpoints(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/Pyramid'):
            os.makedirs('checkpoints/Pyramid')
            if ckpt_path is None:
                ckpt_path = "checkpoints/Pyramid/OC_checkpoint_{}_{}".format(env_name, suffix)
                print(ckpt_path)
            print('Saving models to {}'.format(ckpt_path))
            torch.save({'net': self.net.state_dict(),
                        'prime_net': self.prime_net.state_dict()}, ckpt_path)

    def load_checkpoints(self, path):
        checkpoint = torch.load(path)
        self.net.load_state_dict(torch.load(checkpoint['net']))
        self.prime_net.load_state_dict(torch.load(checkpoint['prime_net']))

        self.net.eval()
        self.prime_net.eval()

    def run(self):
        # return_list = []
        tb_writer = SummaryWriter(log_dir="runs/Pyramid")

        print("Starting Training.")

        episode_rewards = []
        steps_lens = []

        for episode in range(1, self.max_episodes):
            if episode % 2000 == 0:
                self.save_checkpoints(env_name="Pyramids", suffix=episode)

            # if not self.if_unity:
            #     obs = self.env.reset()
            # else:
            #     if not self.if_camera:
            #         obs = list(self.env.reset().values())[0][0][0]  # 84 * 84 * 3
            #     else:
            #         obs = np.array(list(self.env.reset().values())[0][0][0]).reshape(3, 84, 84)

            obs = self.env.reset()[0]
            obs_ = torch.tensor([obs], dtype=torch.float).to(self.device)

            total_reward = 0
            step_number = 0
            episode_num = 0
            loss_total = 0
            greedy_option = self.net.get_option(self.net.get_state(obs_))
            termination = True
            current_option = 0
            for step in range(self.config["max_episode_len"]):
                epsilon = self.epsilon(self.count)

                if termination:
                    current_option = random.choice(
                        list(range(self.n_options))) if epsilon > random.random() else greedy_option

                action, log_prob, entropy = self.net.get_action(
                    self.net.get_state(obs_), current_option)

                next_obs, reward, done, info = self.env.step(action)

                self.count += 1
                total_reward += reward
                step_number += 1

                self.buffer.store(obs, current_option, reward, next_obs[0], done)

                next_obs_ = torch.tensor([next_obs[0]], dtype=torch.float).to(self.device)

                termination, greedy_option = self.net.get_option_termination(
                    self.net.get_state(next_obs_), current_option)

                if len(self.buffer) > self.exploration:
                    loss = 0
                    loss += self.compute_actor_loss(obs_, current_option, log_prob, entropy, reward, done, next_obs_)

                    if self.count % self.update_freq == 0:
                        loss += self.compute_critic_loss()
                        loss_total = loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if self.count % self.freeze_interval == 0:
                        self.prime_net.load_state_dict(self.net.state_dict())

                obs_ = next_obs_

                if done:
                    step_number = step
                    break
                    # if not self.weight_reward:
                    #     self.weight_reward = total_reward
                    # else:
                    #     self.weight_reward = 0.99 * self.weight_reward + 0.01 * total_reward
                    # print(
                    #     'episode: {}  reward: {}  weight_reward: {:.2f}  current_option: {}'.format(i + 1, total_reward,
                    #                                                                                         self.weight_reward,
                    #                                                                             current_option))
            episode_rewards.append(total_reward)
            steps_lens.append(step_number)

            if episode % 10 == 0:
                print("Episode: {:4d}/{}: {}".format(episode, self.config["max_episodes"], np.mean(episode_rewards[-10:])))
                # return_list.append(self.weight_reward)
                tags = ["loss", "reward", "steps"]

                tb_writer.add_scalar(tags[0], loss_total, episode)
                tb_writer.add_scalar(tags[0], np.mean(episode_rewards[-10:]), episode)
                tb_writer.add_scalar(tags[1], np.mean(steps_lens[-10:]), episode)
            # episodes_list = list(range(len(return_list)))
            # plt.plot(episodes_list, return_list)
            # plt.xlabel('Episodes')
            # plt.ylabel('Returns')
            # plt.title('Option-Critic on {}'.format('Unity-FourRooms'))
            # plt.show()

        tb_writer.close()

