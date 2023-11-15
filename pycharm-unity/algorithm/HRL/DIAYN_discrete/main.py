from Unity_Environment.Environments import Pyramids
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

    file_name = None

    p_z = np.full(params["n_skills"], 1 / params["n_skills"])
    agent = SACAgent(p_z=p_z, **params)

    if params["do_train_skill"]:
        env = Pyramids(file_name=file_name, worker_id=0, time_scale=params["train_time_scale"])
        tb_writer = SummaryWriter(log_dir="runs/Pyramid_10skills")
        min_episode = 0
        last_logq_zs = 0
        weight_reward = 0
        # np.random.seed(params["seed"])

        print("Training from scratch.")

        episode_rewards = []
        steps_lens = []

        for episode in range(params["max_n_episodes"]):
            if episode % 2500 == 0:
                agent.save_checkpoint(env_name="Pyramid", suffix=episode)
            z = np.random.choice(params["n_skills"], p=p_z)
            state = env.reset()[0]
            state = concat_state_latent(state, z, params["n_skills"])

            total_reward = 0
            step_number = 1

            for step in range(params["max_episode_len"]):
                action = agent.choose_action(state)

                next_state, reward, done, info = env.step(action)
                next_state = concat_state_latent(next_state[0], z, params["n_skills"])

                total_reward += reward
                step_number += 1

                agent.store(state, z, done, action, next_state)

                agent.train()
                state = next_state

                if done:
                    break

            episode_rewards.append(total_reward)
            steps_lens.append(step_number)

            if episode % 10 == 0:
                print("Episode: {:4d}/{}: {}".format(episode, params["max_episodes"], np.mean(episode_rewards[-10:])))

                tags = ["episode_reward", "steps"]

                tb_writer.add_scalar(tags[0], np.mean(episode_rewards[-10:]), episode)
                tb_writer.add_scalar(tags[1], np.mean(steps_lens[-10:]), episode)

        tb_writer.close()


