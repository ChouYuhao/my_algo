from tqdm import tqdm
import numpy as np
import torch
import collections
import random
from torch.utils.tensorboard import SummaryWriter


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


# def train_on_policy_agent(env, agent, num_episodes):
#     return_list = []
#     tb_writer = SummaryWriter(log_dir="runs/minigird-fourrooms_experiment")
#     for i in range(10):
#         with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
#             for i_episode in range(int(num_episodes / 10)):
#                 episode_return = 0
#                 transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
#                 state = env.reset()
#                 done = False
#                 while not done:
#                     action = agent.take_action(state)
#                     next_state, reward, done, _ = env.step(action)
#                     transition_dict['states'].append(state)
#                     transition_dict['actions'].append(action)
#                     transition_dict['next_states'].append(next_state)
#                     transition_dict['rewards'].append(reward)
#                     transition_dict['dones'].append(done)
#                     state = next_state
#                     episode_return += reward
#                 return_list.append(episode_return)
#
#
#
#                 agent.update(transition_dict)
#                 if (i_episode + 1) % 10 == 0:
#                     pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
#                                       'return': '%.3f' % np.mean(return_list[-10:])})
#                 pbar.update(1)
#
#                 tags = ["reward"]
#                 tb_writer.add_scalar(tags[0], episode_return)
#
#     tb_writer.close()
#
#     return return_list
def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    tb_writer = SummaryWriter(log_dir="runs/cartpole_experiment")
    for i in range(num_episodes):
        # with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        #     for i_episode in range(int(num_episodes / 10)):
        episode_return = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
        print('episode: {}  episode_reward: {}  mean_reward_per100: {:.3f} '.format(i + 1, episode_return,
                                                                                    np.mean(return_list[-100:])))

        agent.update(transition_dict)
        # if (i_episode + 1) % 10 == 0:
        #     pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
        #                       'return': '%.3f' % np.mean(return_list[-10:])})
        # pbar.update(1)

        tags = ["episode_reward", "mean_reward_per100"]
        tb_writer.add_scalar(tags[0], episode_return, i + 1)
        tb_writer.add_scalar(tags[1], np.mean(return_list[-100:]), i + 1)

    tb_writer.close()

    return return_list


# def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
#     return_list = []
#     tb_writer = SummaryWriter(log_dir="runs/minigird-fourrooms_experiment")
#     for i in range(10):
#         with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
#             for i_episode in range(int(num_episodes / 10)):
#                 episode_return = 0
#                 state = env.reset()
#                 done = False
#                 while not done:
#                     action = agent.take_action(state)
#                     next_state, reward, done, info = env.step(action)
#                     replay_buffer.add(state, action, reward, next_state, done)
#                     state = next_state
#                     episode_return += reward
#                     if replay_buffer.size() > minimal_size:
#                         b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
#                         transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
#                                            'dones': b_d}
#                         agent.update(transition_dict)
#                 return_list.append(episode_return)
#                 if (i_episode + 1) % 10 == 0:
#                     pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
#                                       'return': '%.3f' % np.mean(return_list[-10:])})
#                 pbar.update(1)
#
#                 tags = ["reward"]
#                 tb_writer.add_scalar(tags[0], episode_return)
#     tb_writer.close()
#     return return_list
def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    tb_writer = SummaryWriter(log_dir="runs/cartpole_experiment")
    for i in range(num_episodes):
        # with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        #     for i_episode in range(int(num_episodes / 10)):
        episode_return = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                   'dones': b_d}
                agent.update(transition_dict)
        return_list.append(episode_return)
        print('episode: {}  episode_reward: {}  mean_reward_per100: {:.3f} '.format(i + 1, episode_return,
                                                                                    np.mean(return_list[-100:])))
        # if (i_episode + 1) % 10 == 0:
        #     pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
        #                       'return': '%.3f' % np.mean(return_list[-10:])})
        # pbar.update(1)

        tags = ["episode_reward", "mean_reward_per100"]
        tb_writer.add_scalar(tags[0], episode_return, i + 1)
        tb_writer.add_scalar(tags[1], np.mean(return_list[-100:]), i + 1)
    tb_writer.close()

    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
