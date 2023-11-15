import gym
from brain.agent import SACAgent
from Common.play import Play
from Common.config import get_params
import numpy as np

import torch
import unity_wrapper as uw
from torch.utils.tensorboard import SummaryWriter
import os

def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])



if __name__ == "__main__":
    params = get_params()
    # test_env = uw.UnityWrapper(train_mode=params["do_train"],
    #                       # file_name=r'E:\Unity\Envs_master_exe\RLEnvironments.exe',
    #                       # scene='UGV',
    #                       n_agents=1,
    #                       no_graphics=False
    #                       )
    #
    # ma_obs_shapes, ma_d_action_size, ma_c_action_size = test_env.init()

    torch.manual_seed(123)
    # n_states = test_env.observation_space.shape[0]
    # n_actions = test_env.action_space.shape[0]
    # action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]
    # state_0 = list(ma_obs_shapes.values())[0][0][0].reshape(3, 84, 84)
    # state_0 = list(ma_obs_shapes.values())[0][0][0]
    # state_1 = list(ma_obs_shapes.values())[0][2][0]


    # n_states = tuple(list(ma_obs_shapes.values()))[0][2][0]   # {'RosCar?team=0': [(84, 84, 3), (84, 84, 1), (802,), (6,)]}
    # n_states = 1024
    # n_actions = list(ma_c_action_size.values())[0]
    # action_bounds = [-1, 1]
    #
    # params.update({"n_states": n_states,
    #                "n_actions": n_actions,
    #                "action_bounds": action_bounds})
    # print("params:", params)
    #
    # test_env.close()
    # del test_env, n_states, n_actions, action_bounds

    env = uw.UnityWrapper(train_mode=params["do_train_skill"],
                          # file_name=r'E:\Unity\Envs_exe\RLEnvironments.exe',
                          # scene='FourRooms',
                          n_agents=1,
                          no_graphics=True
                          )

    ma_obs_shapes, ma_d_action_size, ma_c_action_size = env.init()
    ma_names = list(ma_obs_shapes.keys())

    p_z = np.full(params["n_skills"], 1 / params["n_skills"])

    agent = SACAgent(p_z=p_z, **params)



    if params["do_train_skill"]:
        tb_writer = SummaryWriter(log_dir="runs/Pyramid_10skills")
        min_episode = 0
        last_logq_zs = 0
        weight_reward = 0
        np.random.seed(params["seed"])
        # env.seed(params["seed"])
        # env.observation_space.seed(params["seed"])
        # env.action_space.seed(params["seed"])
        print("Training from scratch.")

        # for episode in tqdm(range(1 + min_episode, params["max_n_episodes"] + 1)):
        for episode in range(params["max_n_episodes"]):
            if episode % 2500 == 0:
                agent.save_checkpoint(env_name="FourRooms", suffix=episode)
            z = np.random.choice(params["n_skills"], p=p_z)
            # state_1 = np.array(list(env.reset().values())[0][2][0])
            if params["unity_camera"]:
                state_0 = np.array(list(env.reset().values())[0][0][0])
                state_1 = np.array(list(env.reset().values())[0][2][0])
                state = concat_state_latent(agent.check_state(state_0, state_1), z, params["n_skills"])
            else:
                pass
                state = np.array(list(env.reset().values())[0][0][0])
                state = concat_state_latent(state, z, params["n_skills"])
            episode_reward = 0
            logq_zses = []

            # max_n_steps = min(max_episode_len, env.spec.max_episode_steps)
            while True:

                action = agent.choose_action(state)
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
                    next_state = concat_state_latent(agent.check_state(next_state_0, next_state_1), z,
                                                     params["n_skills"])
                else:
                    next_state = np.array(list(env.reset().values())[0][0][0])
                    next_state = concat_state_latent(next_state, z, params["n_skills"])
                reward = list(reward.values())[0][0]
                done = list(done.values())[0][0]
                agent.store(state, z, done, action, next_state)
                logq_zs = agent.train()
                if logq_zs is None:
                    logq_zses.append(last_logq_zs)
                else:
                    logq_zses.append(logq_zs)
                episode_reward += reward
                state = next_state
                if done:
                    print(
                        'episode: {}  logq_zs: {}  episode_reward: {:.2f}  current_skill: {}'.format(episode + 1, logq_zses[-1], episode_reward, z))
                    tags = ["logq_zs", "episode_reward"]

                    tb_writer.add_scalar(tags[0], logq_zses[-1], episode + 1)
                    tb_writer.add_scalar(tags[1], weight_reward, episode + 1)
                    break

        tb_writer.close()

    else:
        agent.load_checkpoint(params["checkpoint_path"])

        player = Play(env, ma_names, ma_d_action_size, ma_c_action_size, agent, n_skills=params["n_skills"], **params)
        player.evaluate_single_skill(5)
