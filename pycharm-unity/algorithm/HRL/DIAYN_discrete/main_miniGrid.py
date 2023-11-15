from minigrid.wrappers import *
from minigrid.wrappers import FlatObsWrapper
from minigrid.wrappers import OneHotPartialObsWrapper
from minigrid.wrappers import ActionBonus
from brain.agent import SACAgent
from Common.config import get_params
from Common.config import get_params
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])


if __name__ == "__main__":
    params = get_params()
    p_z = np.full(params["n_skills"], 1 / params["n_skills"])
    agent = SACAgent(p_z=p_z, **params)

    if params["do_train_skill"]:
        env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="human")
        # env = ActionBonus(env)
        # env = FlatObsWrapper(env)  # obs.dim = 2835

        tb_writer = SummaryWriter(log_dir="runs/DoorKey-5x5")
        # np.random.seed([params["seed"]])
        print("Starting Training.")

        episode_rewards = []
        steps_lens = []

        for episode in range(1, params["max_episodes"]):
            if episode % params['miniGrid_saveTime'] == 0:
                agent.save_checkpoint(env_name="DoorKey-5x5", suffix=episode)
            z = np.random.choice(params["n_skills"], p=p_z)
            obs, info = env.reset()
            obs = concat_state_latent(obs, z, params["n_skills"])
            obs = agent.con_states()
            total_reward = 0
            step_number = 0
            for step in range(env.max_steps):
                action = agent.choose_action(obs)

                next_obs, reward, terminated, truncated, info = env.step(action)
                next_obs = concat_state_latent(next_obs, z, params["n_skills"])

                done = terminated or truncated

                total_reward += reward

                agent.store(obs, z, done, action, next_obs)

                agent.train()
                obs = next_obs

                if done:
                    step_number = step
                    break

            episode_rewards.append(total_reward)
            steps_lens.append(step_number)

            if episode % 10 == 0:
                print("Episode: {:4d}/{}: {}".format(episode, params["max_episodes"], np.mean(episode_rewards[-10:])))

                tags = ["episode_reward", "steps"]

                tb_writer.add_scalar(tags[0], np.mean(episode_rewards[-10:]), episode)
                tb_writer.add_scalar(tags[1], np.mean(steps_lens[-10:]), episode)

        tb_writer.close()