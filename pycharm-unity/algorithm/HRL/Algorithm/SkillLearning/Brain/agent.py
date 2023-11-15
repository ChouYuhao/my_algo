import os
import numpy as np
from .model import FeatureNet, SkillEncode, ActionDecoder, ValueNetWork, QValueNetWork
import torch
from .replay_memory import Memory, Transition
from torch import from_numpy
from torch.optim.adam import Adam
from torch.nn.functional import log_softmax


class ODSEAgent:
    def __init__(self, **config):
        self.config = config
        self.n_states = self.config["n_states"]
        self.n_features = self.config["n_features"]
        self.n_actions = self.config["n_actions"]
        self.n_skills = self.config["n_skills"]
        self.batch_size = self.config["batch_size"]

        self.gamma = self.config["gamma"]
        self.sigma = self.config["sigma"]
        self.tau = self.config["tau"]

        self.memory = Memory(self.config["mem_size"], self.config["seed"])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.feature_network = FeatureNet(n_states=self.n_states, n_features=self.n_features).to(self.device)
        self.skill_encoder = SkillEncode(n_features=self.n_features, n_skills=self.n_skills).to(self.device)
        self.action_decoder = ActionDecoder(n_features=self.n_features + self.n_skills, n_actions=self.n_actions).to(
            self.device)
        self.q_value_network1 = QValueNetWork(n_features=self.n_features + self.n_skills, n_actions=self.n_actions).to(
            self.device)
        self.q_value_network2 = QValueNetWork(n_features=self.n_features + self.n_skills, n_actions=self.n_actions).to(
            self.device)
        self.value_network = ValueNetWork(n_features=self.n_features + self.n_skills).to(self.device)
        self.value_target_network = ValueNetWork(n_features=self.n_features + self.n_skills).to(self.device)

        # self.soft_update_target_network()

        self.encoder_opt = Adam(self.skill_encoder.parameters(), lr=self.config["lr"])
        self.decoder_opt = Adam(self.action_decoder.parameters(), lr=self.config["lr"])
        self.q_value1_opt = Adam(self.q_value_network1.parameters(), lr=self.config["lr"])
        self.q_value2_opt = Adam(self.q_value_network2.parameters(), lr=self.config["lr"])
        self.value_opt = Adam(self.value_network.parameters(), lr=self.config["lr"])

        self.mse_loss = torch.nn.MSELoss()
        self.entropy_loss = torch.nn.CrossEntropyLoss()

    def choose_action(self, feature, skill):
        feature = torch.tensor([feature], dtype=torch.float).to(self.device)
        skill = torch.tensor([skill], dtype=torch.float).to(self.device)
        probs = self.action_decoder(feature, skill)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def soft_update(self, net, target_net):
        for param_target, parma in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1 - self.tau) + parma.data * self.tau)

    def store(self, state, action, reward, next_state, done):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = torch.tensor([action]).to(self.device)
        reward = torch.tensor([reward], dtype=torch.float).to(self.device)
        next_state = torch.tensor([next_state], dtype=torch.float).to(self.device)
        done = torch.BoolTensor([done]).to(self.device)

    def unpack(self, batch):
        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).view(self.batch_size, self.n_states).to(self.device)
        actions = torch.cat(batch.action).view(-1, self.n_actions).to(self.device)
        rewards = torch.cat(batch.reward).view(self.batch_size, 1).to(self.device)
        next_states = torch.cat(batch.next_state).view(self.batch_size, self.n_states).to(self.device)
        dones = torch.cat(batch.done).view(self.batch_size, 1).to(self.device)

        return states, actions, rewards, next_states, dones

    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/Pyramid/ODSE'):
            os.makedirs('checkpoints/Pyramid/ODSE')
        if ckpt_path is None:
            ckpt_path = "checkpoints/Pyramid/ODSE/checkpoint_{}_{}".format(env_name, suffix)
            print(ckpt_path)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({
            'feature_network': self.feature_network.state_dict(),
            'skill_encoder': self.skill_encoder.state_dict(),
            'action_decoder': self.action_decoder.state_dict(),
            'q_value_network1': self.q_value_network1.state_dict(),
            'q_value_network2': self.q_value_network2.state_dict(),
            'value_network': self.value_network.state_dict(),
            'value_target_network': self.value_target_network.state_dict()
        }, ckpt_path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.feature_network.load_state_dict(checkpoint['feature_network'])
        self.skill_encoder.load_state_dict(checkpoint['skill_encoder'])
        self.action_decoder.load_state_dict(checkpoint['action_decoder'])
        self.q_value_network1.load_state_dict(checkpoint['q_value_network1'])
        self.q_value_network2.load_state_dict(checkpoint['q_value_network2'])
        self.value_network.load_state_dict(checkpoint['value_network'])
        self.value_target_network.load_state_dict(checkpoint['value_target_network'])

    def train(self):
        if len(self.memory) < self.batch_size:
            return None
        else:
            batch = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states, dones = self.unpack(batch)

            features = self.feature_network.encoder(states)
            skills = self.skill_encoder(features)
            re_actions = self.action_decoder(features, skills)

            next_q1_value = self.q_value_network1(states, re_actions)
            next_q2_value = self.q_value_network2(states, re_actions)
            next_q_value = torch.min(next_q1_value, next_q2_value)
            target_value = next_q_value.detach() - self.config["alpha"]

            value = self.value_network(features)
            value_loss = self.mse_loss(target_value, value)

            featureNet_loss = self.entropy_loss(self.feature_network(states), states)


