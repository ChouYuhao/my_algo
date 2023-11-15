import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


class FeatureNet(nn.Module):
    def __init__(self, n_states, n_features, n_hidden_filters=256):
        super(FeatureNet, self).__init__()
        self.n_states = n_states
        self.n_features = n_features
        self.n_hidden_filters = n_hidden_filters

        # self.fc1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        # self.fc2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        # self.fc3 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_features)\
        self.encoder = nn.Sequential(
            # nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters),
            # nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_features),
            nn.Linear(in_features=self.n_states, out_features=self.n_features),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.n_features, out_features=self.n_states),
            nn.Sigmoid()
        )

    def forward(self, state):
        # x = self.fc1(state)
        # x = self.fc2(x)
        # return F.relu(self.fc3(x))
        x = self.encoder(state)
        x = self.decoder(state)
        return x

    def get_feature(self, state):
        return self.encoder(state)


class SkillEncode(nn.Module):
    def __init__(self, n_features, n_skills, n_hidden_filters=256):
        super(SkillEncode, self).__init__()
        self.n_skills = n_skills
        self.n_features = n_features
        self.n_hidden_filters = n_hidden_filters

        self.fc1 = nn.Linear(in_features=self.n_features, out_features=self.n_hidden_filters)
        self.fc2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        self.fc3 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_skills)

    def forward(self, n_features):
        x = self.fc1(n_features)
        x = self.fc2(x)
        return F.softmax(self.fc3(x))


# def OnehotEncode(x, dim):  # x表示输入的特征向量,dim表示one-hot向量的长度
#     x = F.softmax(x, dim=1)  # 使用softmax函数对特征向量进行处理，让其每一维值的总和等于1
#     x = torch.argmax(x, dim=1)  # .argmax()函数返回每行最大值的对应下标
#     x_onehot = torch.zeros((x.shape[0], dim))  # 生成one-hot向量
#     x_onehot.scatter_(1, x.unsqueeze(1),
#                       1)  # scatter_()函数将所有维度上除一维外其余维度匹配，将 1 维变为 [seq,1]，使用它来索引和分配数据（将满足条件的数据索引的位上变为 1）
#     return x_onehot


# class SkillEncoder(nn.Module):
#     def __init__(self, n_features, n_skills, n_hidden_filters=256):
#         super(SkillEncoder, self).__init__()
#         self.n_features = n_features
#         self.n_skills = n_skills
#         self.n_hidden_filters = n_hidden_filters
#
#         self.fc1 = nn.Linear(in_features=self.n_features, out_features=self.n_hidden_filters)
#         self.fc2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
#         self.fc3 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_skills)
#
#     def forward(self, feature):
#         x = self.fc1(feature)
#         x = self.fc2(x)
#         x = OnehotEncode(x, dim=self.n_skills)
#         return x

class ActionDecoder(nn.Module):
    def __init__(self, n_features, n_skills, n_actions, n_hidden_filters=256):
        super(ActionDecoder, self).__init__()
        self.n_features = n_features
        self.n_skills = n_skills
        self.n_actions = n_actions
        self.n_hidden_filters = n_hidden_filters

        self.fc1 = nn.Linear(in_features=self.n_features + self.n_skills, out_features=self.n_hidden_filters)
        self.fc2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        self.fc3 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)

    def forward(self, feature, skill):
        x = self.fc1(torch.cat([feature, skill], dim=1))
        x = self.fc2(x)
        return F.softmax(self.fc3(x), dim=1)
        # return probs, action_dist = torch.distributions.Categorical(probs)
        #         action = action_dist.sample()
        #         return action.item()


class ValueNetWork(nn.Module):
    def __init__(self, n_features, n_hidden_filters=256):
        super(ValueNetWork, self).__init__()
        self.n_features = n_features
        self
        self.n_hidden_filters = n_hidden_filters

        self.fc1 = nn.Linear(in_features=self.n_features, out_features=self.n_hidden_filters)
        self.fc2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        self.fc3 = nn.Linear(in_features=self.n_hidden_filters, out_features=1)

    def forward(self, feature):
        x = F.relu(self.fc1(feature))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class QValueNetWork(nn.Module):
    def __init__(self, n_features, n_actions, n_hidden_filters=256):
        super(QValueNetWork, self).__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.n_hidden_filters = n_hidden_filters

        self.fc1 = nn.Linear(in_features=self.n_features + self.n_actions, out_features=self.n_hidden_filters)
        self.fc2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        self.fc3 = nn.Linear(in_features=self.n_hidden_filters, out_features=1)

    def forward(self, feature, action):
        x = torch.cat([feature, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
