from Brain.agent import MultiDDPG_Agent
from Common.config import get_params
import gym
import numpy as np
import unity_wrapper as uw
import os
from torch.utils.tensorboard import SummaryWriter

params = get_params()
env_name = 'Pendulum-v1'
env = gym.make(env_name)
# action_dim = env.action_space.shape[0]
# action_bound_0 = env.action_space.low[0]
# action_bound_1 = env.action_space.high[0]
# state_dim = env.observation_space.shape[0]

agent = MultiDDPG_Agent(**params)

if params["do_train"]:
    tb_writer = SummaryWriter(log_dir="runs/gym/test1_3Actors")
    np.random.seed(params["seed"])
    print("start training...")

    for episode in range(1, 1000 + 1):

        actor_index = np.random.randint(0, 3)

        state = env.reset()
        done = False
        episode_reward = 0
        last_actor_loss = 0
        actor_1_losses = []
        actor_2_losses = []
        actor_3_losses = []
        return_list = []
        while not done:
            action = agent.take_action(state, actor_index)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)
            actor_1_loss, actor_2_loss, actor_3_loss = agent.update()

            if actor_1_loss is None:
                actor_1_losses.append(last_actor_loss)
                actor_2_losses.append(last_actor_loss)
                actor_3_losses.append(last_actor_loss)
            else:
                actor_1_losses.append(actor_1_loss)
                actor_2_losses.append(actor_2_loss)
                actor_3_losses.append(actor_3_loss)

            episode_reward += reward
            state = next_state

        return_list.append(episode_reward)
        print(
            'episode: {}    actor_1_losses: {}    actor_2_losses: {}  actor_3_losses: {}  episode_reward: '
            '{} '.format(
                episode, actor_1_losses[-1], actor_2_losses[-1], actor_3_losses[-1], return_list[-1]))

        tags = ["actor_1_losses", "actor_2_losses", "actor_3_losses", "episode_reward"]
        tb_writer.add_scalar(tags[0], actor_1_losses[-1], episode)
        tb_writer.add_scalar(tags[1], actor_2_losses[-1], episode)
        tb_writer.add_scalar(tags[2], actor_3_losses[-1], episode)
        tb_writer.add_scalar(tags[3], episode_reward, episode)

    tb_writer.close()
