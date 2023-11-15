import random
import torch
from DRL_in_gym_minigird.Algorithm.sac import rl_utils
from sac import SAC
import gym
import numpy as np

# from gym_minigrid.wrappers import *


actor_lr = 1e-3
critic_lr = 1e-2
alpha_lr = 1e-2
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
tau = 0.005  # 软更新参数
buffer_size = 100000
minimal_size = 2500
batch_size = 64
target_entropy = -1
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

# env_name = 'MiniGrid-FourRooms-v0'
env_name = 'CartPole-v0'

env = gym.make(env_name)
# env = FlatObsWrapper(env)

random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = SAC(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr,
            target_entropy, tau, gamma, device)

return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes,
                                              replay_buffer, minimal_size,
                                              batch_size)
