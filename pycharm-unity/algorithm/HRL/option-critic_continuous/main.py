import gym
from replay_buffer import replay_buffer
from net import opt_cri_arch
from OptionCritic import option_critic
import torch
import os


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    cuda = torch.cuda.is_available()
    os.makedirs('./model', exist_ok=True)
    train = option_critic(
        if_unity=False,
        env=env,
        episode=1000,
        exploration=200,
        update_freq=4,
        freeze_interval=200,
        batch_size=32,
        capacity=100000,
        learning_rate=1e-4,
        option_num=2,
        gamma=0.99,
        sigma=0.01,
        termination_reg=0.01,
        epsilon_init=1.,
        decay=10000,
        epsilon_min=0.01,
        entropy_weight=1e-2,
        conv=False,
        cuda=cuda,
        render=False,
        if_camera=False,
        if_train=False,
        save_path='./model/cartpole.pkl'
    )
    train.run()

