from Unity_Environment.Environments import Pyramids
import torch
from torch.utils.tensorboard import SummaryWriter
from agent import SAC_d
from config import get_params
import numpy as np

if __name__ == "__main__":
    params = get_params()

    file_name = None

    agent = SAC_d(**params)

    if params["do_train"]:
        env = Pyramids(file_name=file_name, worker_id=0, time_scale=params["train_time_scale"])
        tb_writer = SummaryWriter(log_dir="runs/Pyramid")
        np.random.seed([params["seed"]])
        print("Starting Training.")

        episode_rewards = []
        steps_lens = []
        for episode in range(1, params["max_episodes"]):
            if episode % 2500 == 0:
                agent.save_checkpoints(env_name="Pyramids", suffix=episode)
            state = env.reset()
            total_reward = 0
            step_number = 0
            for step in range(params["max_episode_len"]):
                action = agent.select_action(state)

                next_state, reward, done, info = env.step(action)

                total_reward += reward
                step_number += 1

                agent.store(state, action, reward, next_state, done)

                agent.update()
                state = next_state

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

