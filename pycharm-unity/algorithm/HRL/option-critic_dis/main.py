from Unity_Environment.Environments import Pyramids
import torch
from torch.utils.tensorboard import SummaryWriter
from OptionCritic import option_critic
from config import get_params
import numpy as np

if __name__ == '__main__':
    params = get_params()

    file_name = None

    env = Pyramids(file_name=file_name, worker_id=0, time_scale=params["train_time_scale"])

    agent = option_critic(env=env, **params)
    agent.run()
