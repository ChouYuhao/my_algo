import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Play:
    def __init__(self, env, ma_names, ma_d_action_size, ma_c_action_size, agent, **config):
        self.config = config
        self.env = env
        self.ma_names = ma_names
        self.ma_d_action_size = ma_d_action_size
        self.ma_c_action_size = ma_c_action_size
        self.agent = agent
        self.episode_reward = []

    def evaluate_Policy(self):
        tb_writer = SummaryWriter(log_dir="play/UGV_experiment/BC/WallStop02")

        s1 = np.array(list(self.env.reset().values())[0][0][0])
        s2 = np.array(list(self.env.reset().values())[0][2][0])
        s3 = np.array(list(self.env.reset().values())[0][3][0])
        s = self.agent.check_state(s1, s2, s3)

        episode_reward = 0
        done = False

        for episode in range(100):
            for _ in range(2000):
                action = self.agent.choose_action(s)
                action = action.reshape(1, 2)
                # action[0][0] 是steer
                # action[0][1] 是motor
                ma_d_action = {}
                ma_c_action = {}
                for n in self.ma_names:
                    d_action, c_action = None, None
                    if self.ma_d_action_size[n]:
                        d_action = action
                        d_action = np.eye(ma_d_action[n], dtype=np.int32)[d_action]
                    if self.ma_c_action_size[n]:
                        c_action = np.zeros((1, self.ma_c_action_size[n]))
                        c_action = action
                    ma_d_action[n] = d_action
                    ma_c_action[n] = c_action

                next_state, reward, done, _ = self.env.step(ma_d_action, ma_c_action)
                if self.config["unity_camera"]:
                    s_1 = np.array(list(next_state.values())[0][0][0])
                    s_2 = np.array(list(next_state.values())[0][2][0])
                    s_3 = np.array(list(next_state.values())[0][3][0])
                    next_state = self.agent.check_state(s_1, s_2, s_3)

                reward = list(reward.values())[0][0]
                done = list(done.values())[0][0]
                episode_reward += reward
                if done:
                    print(f"episode: {episode}, episode_reward: {episode_reward: .4f}")
                    tags = ["episode_reward"]

                    break
                s = next_state





