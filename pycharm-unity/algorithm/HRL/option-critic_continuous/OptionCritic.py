import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from net import opt_cri_arch
from replay_buffer import replay_buffer
import math
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class option_critic(object):
    def __init__(self, if_unity, env, episode, exploration, update_freq, freeze_interval, batch_size, capacity,
                 learning_rate, option_num, gamma, sigma, termination_reg, epsilon_init, decay, epsilon_min,
                 entropy_weight, conv, cuda, render, if_camera, if_train, save_path=None):
        self.env = env
        self.episode = episode
        self.exploration = exploration
        self.update_freq = update_freq
        self.freeze_interval = freeze_interval
        self.batch_size = batch_size
        self.capacity = capacity
        self.learning_rate = learning_rate
        self.option_num = option_num
        self.gamma = gamma
        self.sigma = sigma
        self.termination_reg = termination_reg
        self.epsilon_init = epsilon_init
        self.decay = decay
        self.epsilon_min = epsilon_min
        self.entropy_weight = entropy_weight
        self.conv = conv
        self.cuda = cuda
        self.render = render
        self.if_camera = if_camera
        self.if_train = if_train
        self.save_path = save_path
        self.if_unity = if_unity

        if not self.if_unity:
            if not self.conv:
                self.observation_dim = self.env.observation_space.shape[0]
            else:
                self.observation_dim = self.env.observation_space.shape
            self.action_dim = self.env.action_space.n
            self.epsilon = lambda x: self.epsilon_min + (self.epsilon_init - self.epsilon_min) * math.exp(
                - x / self.decay)

            if if_train:
                self.net = opt_cri_arch(self.observation_dim, self.action_dim, self.option_num, self.conv)
                self.prime_net = opt_cri_arch(self.observation_dim, self.action_dim, self.option_num, self.conv)
                if self.cuda:
                    self.net = self.net.cuda()
                    self.prime_net = self.prime_net.cuda()
                self.prime_net.load_state_dict(self.net.state_dict())
            else:
                self.net = torch.load(self.save_path)
                self.prime_net = torch.load(self.save_path)

            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
            self.buffer = replay_buffer(self.capacity)
            self.count = 0
            self.weight_reward = None

        else:
            self.ma_obs_shapes, self.ma_d_action_size, self.ma_c_action_size = self.env.init()
            self.ma_names = list(self.ma_obs_shapes.keys())
            if not self.conv:
                self.observation_dim = tuple(list(self.ma_obs_shapes.values()))[0][0][0]

            else:
                self.observation_dim = tuple(list(self.ma_obs_shapes.values()))[0][0]

            if list(self.ma_d_action_size.values())[0]:
                self.action_dim = tuple(list(self.ma_d_action_size.values()))[0]
            else:
                self.action_dim = tuple(list(self.ma_c_action_size.values()))[0]

            self.epsilon = lambda x: self.epsilon_min + (self.epsilon_init - self.epsilon_min) * math.exp(
                - x / self.decay)

            if if_train:
                self.net = opt_cri_arch(self.observation_dim, self.action_dim, self.option_num, self.sigma, self.conv)
                self.prime_net = opt_cri_arch(self.observation_dim, self.action_dim, self.option_num, self.sigma,
                                              self.conv)
                if self.cuda:
                    self.net = self.net.cuda()
                    self.prime_net = self.prime_net.cuda()
                self.prime_net.load_state_dict(self.net.state_dict())
            else:
                self.net = torch.load(self.save_path)
                self.prime_net = torch.load(self.save_path)

            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
            self.buffer = replay_buffer(self.capacity)
            self.count = 0
            self.weight_reward = None

    def compute_critic_loss(self, ):
        if self.cuda:
            torch.FloatTensor = torch.cuda.FloatTensor
            torch.LongTensor = torch.cuda.LongTensor

        observations, options, rewards, next_observations, dones = self.buffer.sample(self.batch_size)

        observations = torch.FloatTensor(observations)
        options = torch.LongTensor(options)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_observations = torch.FloatTensor(next_observations)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        if self.if_camera:
            states = self.net.get_state_(observations)
        else:
            states = self.net.get_state(observations)

        q_values = self.net.get_q_value(states)

        if self.if_camera:
            prime_next_states = self.prime_net.get_state_(next_observations)
        else:
            prime_next_states = self.prime_net.get_state(next_observations)

        prime_next_q_values = self.prime_net.get_q_value(prime_next_states)

        if self.if_camera:
            next_states = self.net.get_state_(next_observations)
        else:
            next_states = self.net.get_state(next_observations)

        next_q_values = self.net.get_q_value(next_states)

        next_betas = self.net.get_termination(next_states)
        next_beta = next_betas.gather(1, options.unsqueeze(1))  # unsqueeze(1)在第1维处增加一个维度 (2,3) => (2,1,3)
                                                                # gather(dim, index) 按照index所给的坐标选择元素

        target_q_omega = rewards + self.gamma * (1 - dones) * (
                (1 - next_beta) * prime_next_q_values.gather(1, options.unsqueeze(1)) + next_beta *
                prime_next_q_values.max(1)[0].unsqueeze(1))
        td_error = (target_q_omega.detach() - q_values.gather(1, options.unsqueeze(1))).pow(2).mean()
        return td_error

    def compute_actor_loss(self, obs, option, log_prob, entropy, reward, done, next_obs, ):
        if self.cuda:
            torch.FloatTensor = torch.cuda.FloatTensor
            torch.LongTensor = torch.cuda.LongTensor
        obs = torch.FloatTensor(np.expand_dims(obs, 0))
        next_obs = torch.FloatTensor(np.expand_dims(next_obs, 0))

        state = self.net.get_state(obs)
        next_state = self.net.get_state(next_obs)
        prime_next_state = self.prime_net.get_state(next_obs)

        next_beta = self.net.get_termination(next_state)[:, option]
        beta = self.net.get_termination(state)[:, option]

        q_value = self.net.get_q_value(state)
        next_q_value = self.net.get_q_value(next_state)
        prime_next_q_value = self.prime_net.get_q_value(next_state)

        gt = reward + self.gamma * (1 - done) * (
                (1 - next_beta) * prime_next_q_value[:, option] + next_beta * prime_next_q_value.max(1)[
            0].unsqueeze(0))

        termination_loss = next_beta * (
                (next_q_value[:, option] - next_q_value.max(1)[0].unsqueeze(1)).detach() + self.termination_reg) * (
                                   1 - done)

        policy_loss = -log_prob * (gt - q_value[:, option]).detach() - self.entropy_weight * entropy
        actor_loss = termination_loss + policy_loss
        return actor_loss

    def run(self):
        # return_list = []
        tb_writer = SummaryWriter(log_dir="runs/RosCar_experiment")
        if self.cuda:
            torch.FloatTensor = torch.cuda.FloatTensor
            torch.LongTensor = torch.cuda.LongTensor
        for i in range(self.episode):

            if not self.if_unity:
                obs = self.env.reset()
            else:
                if not self.if_camera:
                    obs = list(self.env.reset().values())[0][0][0]  # 84 * 84 * 3
                else:
                    obs = np.array(list(self.env.reset().values())[0][0][0]).reshape(3, 84, 84)

            if self.render:
                if not self.if_unity:
                    self.env.render()

            total_reward = 0
            episode_num = 0
            loss_total = 0
            greedy_option = self.net.get_option(self.net.get_state(torch.FloatTensor(np.expand_dims(obs, 0))))
            termination = True
            current_option = 0
            while True:
                epsilon = self.epsilon(self.count)
                if termination:
                    current_option = random.choice(
                        list(range(self.option_num))) if epsilon > random.random() else greedy_option

                if not self.if_unity:
                    action, log_prob, entropy = self.net.get_action(
                        self.net.get_state(torch.FloatTensor(np.expand_dims(obs, 0))), current_option)

                    next_obs, reward, done, info = self.env.step(action)
                else:
                    action, log_prob, entropy = self.net.get_continuous_action(
                        self.net.get_state(torch.FloatTensor(np.expand_dims(obs, 0))), current_option)

                    ma_d_action = {}
                    ma_c_action = {}
                    for n in self.ma_names:
                        d_action, c_action = None, None
                        if self.ma_d_action_size[n]:
                            d_action = action
                            d_action = np.eye(self.ma_d_action_size[n], dtype=np.int32)[d_action]
                        if self.ma_c_action_size[n]:
                            c_action = np.zeros((1, self.ma_c_action_size[n]))
                            c_action = action

                    ma_d_action[n] = d_action
                    ma_c_action[n] = c_action

                    next_obs, reward, done, info = self.env.step(ma_d_action, ma_c_action)

                    if self.if_camera:
                        next_obs = np.array(list(next_obs.values())[0][0][0]).reshape(3, 84, 84)
                    else:
                        next_obs = list(next_obs.values())[0][0][0]

                    reward = list(reward.values())[0][0]
                    done = list(done.values())[0][0]

                self.count += 1
                total_reward += reward
                self.buffer.store(obs, current_option, reward, next_obs, done)

                if self.render:
                    if not self.if_unity:
                        self.env.render()

                termination, greedy_option = self.net.get_option_termination(
                    self.net.get_state(torch.FloatTensor(np.expand_dims(next_obs, 0))), current_option)

                if len(self.buffer) > self.exploration:
                    loss = 0
                    loss += self.compute_actor_loss(obs, current_option, log_prob, entropy, reward, done, next_obs)

                    if self.count % self.update_freq == 0:
                        loss += self.compute_critic_loss()
                        loss_total = loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if self.count % self.freeze_interval == 0:
                        self.prime_net.load_state_dict(self.net.state_dict())

                obs = next_obs

                if done:
                    if not self.weight_reward:
                        self.weight_reward = total_reward
                    else:
                        self.weight_reward = 0.99 * self.weight_reward + 0.01 * total_reward
                    print(
                        'episode: {}  reward: {}  weight_reward: {:.2f}  current_option: {}'.format(i + 1, total_reward,
                                                                                                    self.weight_reward,
                                                                                                    current_option))

                    # return_list.append(self.weight_reward)
                    tags = ["loss", "reward", "weight_reward"]

                    tb_writer.add_scalar(tags[0], loss_total, i + 1)
                    tb_writer.add_scalar(tags[1], total_reward, i + 1)
                    tb_writer.add_scalar(tags[2], self.weight_reward, i + 1)
                    break

            # episodes_list = list(range(len(return_list)))
            # plt.plot(episodes_list, return_list)
            # plt.xlabel('Episodes')
            # plt.ylabel('Returns')
            # plt.title('Option-Critic on {}'.format('Unity-FourRooms'))
            # plt.show()

        tb_writer.close()

        if self.save_path:
            torch.save(self.net, self.save_path)
