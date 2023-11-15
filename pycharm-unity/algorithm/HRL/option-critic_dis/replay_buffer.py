import numpy as np
import random
from collections import deque
import torch

class replay_buffer(object):
    def __init__(self, capacity):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def store(self, observation, option, reward, next_observation, done):
        # observation = np.expand_dims(observation, 0)
        # next_observation = np.expand_dims(next_observation, 0)
        self.memory.append([observation, option, reward, next_observation, done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observation, option, reward, next_observation, done = zip(* batch)

        observation = torch.tensor([observation], dtype=torch.float).to(self.device)
        option = torch.tensor([option]).to(self.device)
        reward = torch.tensor([reward], dtype=torch.float).to(self.device)
        next_observation = torch.tensor([next_observation], dtype=torch.float).to(self.device)
        done = torch.tensor([done]).to(self.device)

        return observation, option, reward, next_observation, done

    def __len__(self):
        return len(self.memory)
