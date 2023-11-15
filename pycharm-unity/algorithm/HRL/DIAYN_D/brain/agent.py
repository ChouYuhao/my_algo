import os
import numpy as np
from .model import PolicyNetwork, QvalueNetwork, ValueNetwork, Discriminator, CNNFeatureNet
import torch
from .replay_memory import Memory, Transition
from torch import from_numpy
from torch.optim.adam import Adam
from torch.nn.functional import log_softmax

class SACAgent:
    def __init__(self,
                 p_z,
                 **config):
        self.config = config
        self.obs_dim = self.config["obs_dim"]
        self.n_states = self.config["n_states"]
        self.n_features = self.config["n_features"]
        self.n_skills = self.config["n_skills"]
        self.batch_size = self.config["batch_size"]
        self.hidden_channels = self.config["hidden_channels"]
        # self.if_train = if_train
        self.p_z = np.tile(p_z, self.batch_size).reshape(self.batch_size, self.n_skills)
        self.memory = Memory(self.config["mem_size"], self.config["seed"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.conv = self.config["if_conv"]
        self.hidden_channels = self.config["hidden_channels"]
        self.n_features = self.config["n_features"]

        if self.conv:
            self.cnnFeatureNet = CNNFeatureNet(in_channels=3, hidden_channels=self.hidden_channels,
                                               out_dim=self.n_features).to(self.device)
            self.n_states = self.n_features

        self.policy_network = PolicyNetwork(n_features=self.n_states + self.n_skills,
                                            n_actions=self.config["n_actions"],).to(self.device)

        self.q_value_network1 = QvalueNetwork(n_features=self.n_states + self.n_skills,).to(self.device)

        self.q_value_network2 = QvalueNetwork(n_features=self.n_states + self.n_skills,).to(self.device)

        self.value_network = ValueNetwork(n_features=self.n_states + self.n_skills).to(self.device)

        self.value_target_network = ValueNetwork(n_features=self.n_states + self.n_skills).to(self.device)

        self.hard_update_target_network()

        self.discriminator = Discriminator(n_features=self.n_states, n_skills=self.n_skills).to(self.device)

        self.mse_loss = torch.nn.MSELoss()
        self.cross_ent_loss = torch.nn.CrossEntropyLoss()

        self.value_opt = Adam(self.value_network.parameters(), lr=self.config["lr"])
        self.q_value1_opt = Adam(self.q_value_network1.parameters(), lr=self.config["lr"])
        self.q_value2_opt = Adam(self.q_value_network2.parameters(), lr=self.config["lr"])
        self.policy_opt = Adam(self.policy_network.parameters(), lr=self.config["lr"])
        self.discriminator_opt = Adam(self.discriminator.parameters(), lr=self.config["lr"])

        self.cnnFeatureNet = CNNFeatureNet(in_channels=3, hidden_channels=self.hidden_channels, out_dim=self.n_features).to(self.device)

    def con_states(self, obs, z_, n):
        obs = torch.tensor(np.array(obs['image']), dtype=torch.float).to(self.device)
        obs = torch.transpose(obs, 0, 2)
        obs = torch.unsqueeze(obs, dim=0)

        states = self.cnnFeatureNet.get_states(obs).cpu().numpy()

        z_one_hot = np.zeros(n)
        z_one_hot[z_] = 1
        return np.concatenate([states[0], z_one_hot])

    def choose_action(self, states):
        states = np.expand_dims(states, axis=0)
        states = from_numpy(states).float().to(self.device)
        action, _ = self.policy_network.sample_or_likelihood(states)
        return action.item()

    def store(self, state, z, done, action, next_state):
        state = from_numpy(state).float().to("cpu")
        next_state = from_numpy(next_state).float().to("cpu")

        z = torch.ByteTensor([z]).to("cpu")
        done = torch.BoolTensor([done]).to("cpu")
        action = torch.Tensor([action]).to("cpu")

        # state = from_numpy(state).float().to(self.device)
        # z = torch.ByteTensor([z]).to(self.device)
        # done = torch.BoolTensor([done]).to(self.device)
        # action = torch.Tensor([action]).to(self.device)
        # next_state = from_numpy(next_state).float().to(self.device)
        self.memory.add(state, z, done, action, next_state)

    def unpack(self, batch):
        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).view(self.batch_size, self.n_states + self.n_skills).to(self.device)
        next_states = torch.cat(batch.next_state).view(self.batch_size, self.n_states + self.n_skills).to(
                self.device)
        zs = torch.cat(batch.z).view(self.batch_size, 1).long().to(self.device)
        dones = torch.cat(batch.done).view(self.batch_size, 1).to(self.device)
        actions = torch.cat(batch.action).view(self.batch_size, 1).to(self.device)


        return states, zs, dones, actions, next_states

    def train(self):
        if len(self.memory) < self.batch_size:
            return None
        else:
            batch = self.memory.sample(self.batch_size)
            states, zs, dones, actions, next_states = self.unpack(batch)

            # states = self.cnnFeatureNet(states)
            # next_states = self.cnnFeatureNet(next_states)

            p_z = from_numpy(self.p_z).to(self.device)

            # Calculating the value target
            reparam_actions, log_probs = self.policy_network.sample_or_likelihood(states)
            q1 = self.q_value_network1(states, reparam_actions)
            q2 = self.q_value_network2(states, reparam_actions)
            q = torch.min(q1, q2)
            target_value = q.detach() - self.config["alpha"] * log_probs.detach()

            value = self.value_network(states)
            value_loss = self.mse_loss(value, target_value)

            logits = self.discriminator(torch.split(next_states, [self.n_features, self.n_skills], dim=-1)[0])
            p_z = p_z.gather(-1, zs)
            logq_z_ns = log_softmax(logits, dim=-1)
            rewards = logq_z_ns.gather(-1, zs).detach() - torch.log(p_z + 1e-6)

            # Calculating the Q-Value target
            with torch.no_grad():
                target_q = self.config["reward_scale"] * rewards.float() + \
                           self.config["gamma"] * self.value_target_network(next_states) * (~dones)
            q1 = self.q_value_network1(states, actions.squeeze(-1))
            q2 = self.q_value_network2(states, actions.squeeze(-1))
            q1_loss = self.mse_loss(q1, target_q)
            q2_loss = self.mse_loss(q2, target_q)

            policy_loss = (self.config["alpha"] * log_probs - q).mean()
            logits = self.discriminator(torch.split(states, [self.n_features, self.n_skills], dim=-1)[0])
            discriminator_loss = self.cross_ent_loss(logits, zs.squeeze(-1))

            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            self.value_opt.zero_grad()
            value_loss.backward()
            self.value_opt.step()

            self.q_value1_opt.zero_grad()
            q1_loss.backward()
            self.q_value1_opt.step()

            self.q_value2_opt.zero_grad()
            q2_loss.backward()
            self.q_value2_opt.step()

            self.discriminator_opt.zero_grad()
            discriminator_loss.backward()
            self.discriminator_opt.step()

            self.soft_update_target_network(self.value_network, self.value_target_network)

            return -discriminator_loss.item()

    def soft_update_target_network(self, local_network, target_network):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(self.config["tau"] * local_param.data +
                                    (1 - self.config["tau"]) * target_param.data)

    def hard_update_target_network(self):
        self.value_target_network.load_state_dict(self.value_network.state_dict())
        self.value_target_network.eval()

    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):

        if not os.path.exists('checkpoints/DoorKey-5x5/image'):
            os.makedirs('checkpoints/DoorKey-5x5/image')
        if ckpt_path is None:
            ckpt_path = "checkpoints/DoorKey-5x5/image/DIAYN_checkpoint_{}_{}".format(env_name, suffix)
            print(ckpt_path)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_network': self.policy_network.state_dict(),
                    'q_value_network1': self.q_value_network1.state_dict(),
                    'q_value_network2': self.q_value_network2.state_dict(),
                    'value_network': self.value_network.state_dict(),
                    'value_target_network': self.value_target_network.state_dict(),
                    'discriminator': self.discriminator.state_dict()}, ckpt_path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.q_value_network1.load_state_dict(checkpoint['q_value_network1'])
        self.q_value_network2.load_state_dict(checkpoint['q_value_network2'])
        self.value_network.load_state_dict(checkpoint['value_network'])
        self.value_target_network.load_state_dict(checkpoint['value_target_network'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])

        self.policy_network.eval()
        self.q_value_network1.eval()
        self.q_value_network2.eval()
        self.value_network.eval()
        self.value_target_network.eval()
        self.discriminator.eval()


