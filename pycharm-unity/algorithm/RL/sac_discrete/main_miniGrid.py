import gymnasium as gym
from minigrid.wrappers import *
from minigrid.wrappers import FlatObsWrapper
from minigrid.wrappers import RGBImgObsWrapper
from minigrid.wrappers import OneHotPartialObsWrapper
from minigrid.wrappers import SymbolicObsWrapper
from minigrid.wrappers import ActionBonus
from torch.utils.tensorboard import SummaryWriter
from agent import SAC_d
from config import get_params
import numpy as np

# LunarLander-v2
# MiniGrid-Empty-5x5-v0

# env = RGBImgPartialObsWrapper(env)
# env = OneHotPartialObsWrapper(env)


if __name__ == "__main__":
    params = get_params()

    agent = SAC_d(**params)

    if params["do_train"]:
        env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")
        # env = gym.make("MiniGrid-LavaCrossingS9N1-v0", render_mode="human")
        # env = gym.make("CartPole-v1", render_mode="human")
        # env = ActionBonus(env)
        if not params['if_conv']:
            # env = FlatObsWrapper(env)  # obs.dim = 2835
            env = SymbolicObsWrapper(env)
        else:
            env = RGBImgObsWrapper(env)  # 40 * 40 *3

        if params["if_conv"]:
            tb_writer = SummaryWriter(log_dir="runs/empty")
        else:
            tb_writer = SummaryWriter(log_dir="runs/empty")
        # np.random.seed([params["seed"]])
        print("Starting Training.")

        episode_rewards = []
        steps_lens = []
        for episode in range(1, params["max_episodes"]):
            if episode % params['miniGrid_saveTime'] == 0:
                agent.save_checkpoints(env_name="LavaCrossing", suffix=episode)
            obs, info = env.reset()
            total_reward = 0
            step_number = 0
            for step in range(env.max_steps):

                action = agent.select_action(obs)

                next_obs, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated

                total_reward += reward

                agent.store(obs, action, reward, next_obs, done)

                agent.update()
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
