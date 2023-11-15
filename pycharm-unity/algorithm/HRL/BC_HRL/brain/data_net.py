import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 1), stride=(1, 1))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), stride=(1, 1))
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(7056, 512)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.relu3(self.conv3(x))
        x = x.reshape(1, -1)
        x = self.fc(x)
        return x


class Ray(nn.Module):
    def __init__(self):
        super(Ray, self).__init__()
        self.fc1 = nn.Linear(802, 512)
        self.fc2 = nn.Linear(512, 512)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x = self.tanh(self.fc2(self.fc1(x)))
        x = self.tanh(self.fc2(self.fc1(x)))
        return self.tanh(x)
