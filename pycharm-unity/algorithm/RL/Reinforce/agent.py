import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from net import PolicyNet, CNNFeatureNet
from replay_memory import Transition, Memory


class REINFORCEMENT:
    def __init__(self, **config):
        self.config = config
