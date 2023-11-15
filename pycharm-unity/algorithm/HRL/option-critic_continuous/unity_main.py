import gym
import unity_wrapper as uw
from replay_buffer import replay_buffer
from net import opt_cri_arch
from OptionCritic import option_critic
import torch
import os


if __name__ == '__main__':
    env = uw.UnityWrapper(train_mode=True,
                          # file_name=r'E:\Unity\Envs_exe\RLEnvironments.exe',
                          # scene='RosCar',
                          n_agents=1,
                          no_graphics=False
                          )



    cuda = torch.cuda.is_available()
    os.makedirs('./model', exist_ok=True)
    train = option_critic(
        if_unity=True,
        env=env,
        episode=10000,
        exploration=10000,
        update_freq=4,
        freeze_interval=200,
        batch_size=32,
        capacity=100000,
        learning_rate=1e-4,
        option_num=8,
        gamma=0.99,
        sigma=0.01,
        termination_reg=0.01,
        epsilon_init=1.,
        decay=10000,
        epsilon_min=0.01,
        entropy_weight=1e-2,
        conv=True,
        cuda=cuda,
        render=False,
        if_camera=True,
        if_train=True,
        save_path='./model/unity_RosCar.pkl'

    )
    train.run()
