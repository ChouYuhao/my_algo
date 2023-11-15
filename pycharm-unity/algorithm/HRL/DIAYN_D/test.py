from minigrid.wrappers import *
import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper
import matplotlib.pyplot as plt
import torch

env = gym.make("MiniGrid-DoorKey-5x5-v0")
env = RGBImgObsWrapper(env)

obs, _ = env.reset()
# image = 40 * 40 * 3



