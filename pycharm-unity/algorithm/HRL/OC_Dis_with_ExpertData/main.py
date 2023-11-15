from Unity_Environment.Environments import Pyramids
import torch
from torch.utils.tensorboard import SummaryWriter
from agent import OptionCriticAgent
from config import get_params
import numpy as np
import gym

if __name__ == "__main__":
    params = get_params()
    torch.manual_seed(123)

    file_name = None
    # if params["do_train"]:
    #     time_scale = params["train_time_scale"]
    # else:
    #     time_scale = params["play_time_scale"]
    agent = OptionCriticAgent(**params)

    if params["do_train"]:
        # env = Pyramids(file_name=file_name, worker_id=0, time_scale=params["train_time_scale"])
        # env = gym.make('CartPole-v1', render_mode="human")
        tb_writer = SummaryWriter(log_dir="runs/Pyramid")
        np.random.seed([params["seed"]])
        print("Starting Training.")

        # episode_rewards = []
        # steps_lens = []
        total_loss = []

        for episode in range(1, params["max_episodes"]):
            if episode % 2500 == 0:
                agent.save_checkpoints(env_name="Pyramid", suffix=episode)

            # state = env.reset()[0]
            # state = env.reset()
            # env.render()
            # total_reward = 0
            # step_number = 1
            loss = 0
            for step in range(params["max_episode_len"]):
                # option, action = agent.select_action(state)

                # next_state, reward, done, info = env.step(action)
                # next_state, reward, done, info, _ = env.step(action)

                # total_reward += reward
                # step_number += 1

                loss = agent.update()
                # state = next_state

                # if done:
                #     break

            # episode_rewards.append(total_reward)
            # steps_lens.append(step_number)

            total_loss.append(loss)

            if episode % 10 == 0:
                print("Episode: {:4d}/{}: {}".format(episode, params["max_episodes"], np.mean(loss[-10:])))

                tags = ["loss", "steps"]

                tb_writer.add_scalar(tags[0], np.mean(loss[-10:]), episode)
                # tb_writer.add_scalar(tags[1], np.mean(steps_lens[-10:]), episode)

        tb_writer.close()





