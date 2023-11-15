import random
import os
import numpy as np
from .bc_model import PolicyNetwork, CNN, Ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from mlagents.trainers import demo_loader
from torch import from_numpy


class BC_Agent:
    def __init__(self, **config):
        self.config = config
        self.n_states = self.config["n_states"]
        self.n_actions = self.config["n_actions"]
        self.batch_size = self.config["batch_size"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        torch.manual_seed(self.config["seed"])
        self.policy_net = PolicyNetwork(n_states=self.n_states,
                                        n_actions=self.n_actions,
                                        action_bounds=self.config["action_bounds"]).to(self.device)

        self.policy_net_opt = torch.optim.Adam(self.policy_net.parameters(), lr=self.config["lr"])

        self.loss_fc = nn.MSELoss()

        self.demo_path = self.config["demo_path"]
        self.pkl_exist = False
        self.expert_buffer = None
        self.file = None

        if self.config["unity_camera"]:
            self.cnn = CNN().to(self.device)
            self.ray = Ray().to(self.device)

    def unpack_expertData(self, batch_size):
        expert_batch = random.sample(self.expert_buffer, batch_size)
        expert_batch = np.vstack(expert_batch)
        expert_batch = torch.FloatTensor(expert_batch).to(self.device)
        return expert_batch[:, :self.config["n_states"]], expert_batch[:, self.config["n_states"]:]

    def update(self):
        if self.pkl_exist:
            self.file = open(self.config["pkl_filePath"], 'rb')
            self.expert_buffer = pickle.load(self.file)
        expert_states, expert_actions = self.unpack_expertData(batch_size=self.config["batch_size"])
        actions_predict = self.policy_net(expert_states)
        loss = self.loss_fc(actions_predict, expert_actions)

        self.policy_net_opt.zero_grad()
        loss.backward()
        self.policy_net_opt.step()

        return loss

    def save_checkpoints(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/BC_HRL/WallStop02'):
            os.makedirs('checkpoints/BC_HRL/WallStop02')
        if ckpt_path is None:
            ckpt_path = "checkpoints/BC_HRL/WallStop02/checkpoint_{}_{}".format(env_name, suffix)
            print(ckpt_path)
        print('Saving models to {}'.format(ckpt_path))
        if self.config["unity_camera"]:
            torch.save({'policy_net': self.policy_net.state_dict(),
                        'CNN': self.cnn.state_dict(),
                        'Ray': self.ray.state_dict()}, ckpt_path)
        else:
            torch.save({'policy_net': self.policy_net.state_dict()}, ckpt_path)

    def load_checkpoints(self, path):
        checkpoints = torch.load(path)
        self.policy_net.load_state_dict(checkpoints['policy_net'])
        self.policy_net.eval()

        if self.config["unity_camera"]:
            self.cnn.load_state_dict(checkpoints['CNN'])
            self.ray.load_state_dict(checkpoints['Ray'])
            self.cnn.eval()
            self.ray.eval()

    def check_state(self, state_0, state_1, state_2):
        image = torch.unsqueeze(torch.tensor(state_0, dtype=torch.float), 0).to(self.device)
        ray = torch.unsqueeze(torch.tensor(state_1, dtype=torch.float), 0).to(self.device)

        image = self.cnn(image).detach().cpu().numpy().squeeze()
        ray = self.ray(ray).detach().cpu().numpy().squeeze()

        state = np.concatenate((image, ray), axis=0)

        state = np.concatenate((state, state_2), axis=0)
        return state

    def choose_action(self, states):
        states = np.expand_dims(states, axis=0)
        states = from_numpy(states).float().to(self.device)
        action = self.policy_net(states)
        return action.detach().cpu().numpy()[0]

    # 对unity_demo中的数据进行处理，并且通过cnn，ray网络映射
    def LoadDataFromUnityDemo(self):
        _, info_action_pairs, _ = demo_loader.load_demonstration(self.demo_path)

        # TODO 对于未使用unity相机等传感器的状态收集？
        observations = []
        next_observations = []

        # agent_info.observations[0] 维度为 84*84*3 相机图像信息
        # agent_info.observations[1] 维度为 84*84*1 深度相机图像信息
        # agent_info.observations[2] 维度为 802 雷达信息
        # agent_info.observations[3] 维度为 6 位置速度等信息

        # 处理图像信息 并经过卷积
        obs_pre_0 = np.reshape(np.array(info_action_pairs[0].agent_info.observations[0].float_data.data,
                                        dtype=np.float32), (84, 84, 3))

        # 处理雷达信息 并经过MLP
        obs_pre_1 = np.array(info_action_pairs[0].agent_info.observations[2].float_data.data,
                             dtype=np.float32)

        # 位置速度等信息无需处理
        obs_pre_2 = np.array(info_action_pairs[0].agent_info.observations[3].float_data.data,
                             dtype=np.float32)

        # 拼接成最后的观测信息
        # obs_pre = np.concatenate(
        #     (np.concatenate((obs_pre_0, obs_pre_1), axis=0), obs_pre_2), axis=0)
        obs_pre = self.check_state(obs_pre_0, obs_pre_1, obs_pre_2)
        done_pre = info_action_pairs[0].agent_info.done

        actions = []
        rewards = []
        dones = []

        # 轨迹库
        traj_pool = []

        for info_action_pair in info_action_pairs[1:]:
            agent_info = info_action_pair.agent_info
            action_info = info_action_pair.action_info

            # 每个episode数据
            episode_traj = []

            # 处理图像信息 并经过卷积
            obs_0 = np.reshape(np.array(agent_info.observations[0].float_data.data, dtype=np.float32), (84, 84, 3))

            # 处理雷达信息 并经过线性层
            obs_1 = np.array(agent_info.observations[2].float_data.data, dtype=np.float32)

            # 位置速度等信息无需处理
            obs_2 = np.array(agent_info.observations[3].float_data.data, dtype=np.float32)

            obs = self.check_state(obs_0, obs_1, obs_2)

            rew = agent_info.reward
            act = np.array(action_info.continuous_actions, dtype=np.float32)
            done = agent_info.done
            if not done_pre:
                observations.append(obs_pre)
                actions.append(act)
                rewards.append(rew)
                dones.append(done)
                next_observations.append(obs)

                # 存储 (s, a)对
                s_a = np.append(obs_pre, act)
                episode_traj.extend(([s_a]))

            obs_pre = obs
            done_pre = done

            traj_pool.extend(episode_traj)
            del episode_traj[:]

        file = open(self.config["pkl_filePath"], 'wb')
        pickle.dump(traj_pool, file)
        # file.close()
        data = dict(
            obs=np.array(observations, dtype=np.float32),
            acts=np.array(actions, dtype=np.float32),
            rews=np.array(rewards, dtype=np.float32),
            next_obs=np.array(next_observations, dtype=np.float32),
            done=np.array(dones, dtype=np.float32),
        )
        self.pkl_exist = True
        print("demo数据处理完毕.")
        return data
