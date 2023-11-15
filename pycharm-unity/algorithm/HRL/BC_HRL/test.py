import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import gym
from brain.bc_agent import BC_Agent
# from Common.play import Play
from Common.config import get_params

import torch
import unity_wrapper as uw
from torch.utils.tensorboard import SummaryWriter
import os

# class Test_Net(nn.Module):
#     def __init__(self):
#         super(Test_Net, self).__init__()
#         self.fc1 = nn.Linear(10, 8)
#         self.fc2 = nn.Linear(8, 4)
#
#     def forward(self, x):
#         x = self.fc2(self.fc1(x))
#         return x


if __name__ == '__main__':
    # test_net = Test_Net()
    # test_data = np.random.randn(2, 10)
    # test_data = torch.Tensor(test_data)
    # for i in range(2):
    #     # print(test_data)
    #     print(test_net(test_data))
    params = get_params()
    agent = BC_Agent(**params)
    demo_path = "E:\\705(3)\Paper\experience\ml-agents-develop\Project\Assets\ML-Agents\Examples\Pyramids\Demos/ExpertPyramid.demo"
    print(agent.LoadDataFromUnityDemo(demo_path))
