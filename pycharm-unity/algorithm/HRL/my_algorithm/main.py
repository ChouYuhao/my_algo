from brain.H_agent import HRL_Agent
from Common.config import get_params
from Common.play import Play
import numpy as np
import unity_wrapper as uw
import os
from torch.utils.tensorboard import SummaryWriter


def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])


if __name__ == "__main__":
    params = get_params()

    env = uw.UnityWrapper(train_mode=params["do_train"],
                          # file_name=r'E:\Unity\Envs_exe\RLEnvironments.exe',
                          # scene='AutoRooms',
                          n_agents=1,
                          no_graphics=False
                          # use camera no_graphics=False, else True
                          )
    ma_obs_shapes, ma_d_action_size, ma_c_action_size = env.init()
    ma_names = list(ma_obs_shapes.keys())

    p_z = np.full(params["n_skills"], 1 / params["n_skills"])

    agent = HRL_Agent(p_z=p_z, **params)

    if params["do_train"]:
        tb_writer = SummaryWriter(log_dir="runs/AutoRooms_experiment/test_1_10skills")
        # last_log_zs = 0
        last_H_actor_loss = 0
        last_L_actor_loss = 0
        H_actor_losses = []
        L_actor_losses = []
        np.random.seed(params["seed"])
        print("start training...")

        for episode in range(1, params["n_episode"] + 1):
            if (episode - 1) % 2500 == 0:
                agent.save_checkpoints(env_name="UGV", suffix=episode)
            if params["unity_camera"]:
                state_0 = np.array(list(env.reset().values())[0][0][0])
                state_1 = np.array(list(env.reset().values())[0][2][0])
                L_state = agent.check_state(state_0, state_1)
            else:
                L_state = np.array(list(env.reset().values())[0][0][0])
            z = agent.take_skill(L_state)
            H_state = concat_state_latent(L_state, z, params["n_skills"])
            episode_reward = 0
            termination = False
            while True:
                if termination:
                    z = agent.take_skill(L_state)
                    H_state = concat_state_latent(L_state, z, params["n_skills"])
                action = agent.take_action(H_state)
                action = action.reshape(1, 2)

                ma_d_action = {}
                ma_c_action = {}
                for n in ma_names:
                    d_action, c_action = None, None
                    if ma_d_action_size[n]:
                        d_action = action
                        d_action = np.eye(ma_d_action_size[n], dtype=np.int32)[d_action]
                    if ma_c_action_size[n]:
                        c_action = np.zeros((1, ma_c_action_size[n]))
                        c_action = action

                    ma_d_action[n] = d_action
                    ma_c_action[n] = c_action

                next_state, reward, done, _ = env.step(ma_d_action, ma_c_action)
                if params["unity_camera"]:
                    next_state_0 = np.array(list(next_state.values())[0][0][0])
                    next_state_1 = np.array(list(next_state.values())[0][2][0])
                    next_L_state = agent.check_state(next_state_0, next_state_1)
                    next_H_state = concat_state_latent(next_L_state, z, params["n_skills"])
                else:
                    next_L_state = np.array(list(env.reset().values())[0][0][0])
                    next_H_state = concat_state_latent(next_L_state, z, params["n_skills"])
                reward = np.array(list(reward.values())[0] * params["reward_scale"])
                done = np.array(list(done.values())[0])
                agent.store(H_state, z, action, reward, next_H_state, done)

                termination = agent.get_termination(next_L_state, z)

                H_actor_loss, L_actor_loss = agent.update()

                if H_actor_loss is None:
                    H_actor_losses.append(last_H_actor_loss)
                else:
                    H_actor_losses.append(H_actor_loss)
                if L_actor_loss is None:
                    L_actor_losses.append(last_L_actor_loss)
                else:
                    L_actor_losses.append(L_actor_loss)

                episode_reward += reward
                L_state = next_L_state
                H_state = next_H_state

                if done:
                    print(
                        'episode: {}    H_actor_loss: {}    L_actor_loss: {}    episode_reward: {}'.format(
                            episode, H_actor_losses[-1], L_actor_losses[-1], episode_reward))

                    tags = ["H_actor_loss", "L_actor_loss", "episode_reward"]
                    tb_writer.add_scalar(tags[0], H_actor_losses[-1], episode)
                    tb_writer.add_scalar(tags[1], L_actor_losses[-1], episode)
                    tb_writer.add_scalar(tags[2], episode_reward, episode)
                    break

        tb_writer.close()

    else:
        agent.load_checkpoints(params["checkpoints_path"])
        player = Play(env, ma_names, ma_d_action_size, ma_c_action_size, agent, **params)
        player.agent_eval()

