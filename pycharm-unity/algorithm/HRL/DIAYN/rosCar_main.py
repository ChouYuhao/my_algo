import gym
from brain.agent import SACAgent
from Common.play import Play
from Common.config import get_params
import numpy as np
import torch
from ros_ugv import *
import time


def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])


if __name__ == "__main__":
    params = get_params()
    car = RosUGV()
    while not car.initialized():
        time.sleep(0.5)
    print("car initialized")
    print('laser_scan_size', len(car.laser_scan))
    if LASER_SCAN_SIZE != len(car.laser_scan):
        print(f'warning! laser_scan_size != {LASER_SCAN_SIZE}')

    p_z = np.full(params["n_skills"], 1 / params["n_skills"])

    agent = SACAgent(p_z=p_z, **params)
    agent.load_checkpoint(params["checkpoint_path"])

    while not car.is_shutdown():
        state = car.get_rl_obs_list()
        state_0 = state[0]
        state_1 = state[2]
        state = concat_state_latent(agent.check_state(state_0, state_1), 5, params["n_skills"])
        # next_state = concat_state_latent(agent.check_state(next_state_0, next_state_1), z, params["n_skills"])
        action = agent.choose_action(state)
        car.move_ugv(0.1 * action[0][1], action[0][0])

