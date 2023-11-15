from Brain.agent import MultiDDPG_Agent
from Common.config import get_params

import numpy as np
import unity_wrapper as uw
import os
from torch.utils.tensorboard import SummaryWriter

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

    agent = MultiDDPG_Agent(**params)

    if params["do_train"]:
        tb_writer = SummaryWriter(log_dir="runs/UGV/test1_3Actors")
        np.random.seed(params["seed"])
        print("start training...")

        for episode in range(1, params["n_episode"] + 1):
            if (episode - 1) % 2500 == 0:
                agent.save_checkpoints(env_name="UGV", suffix=episode)
            actor_index = np.random.randint(0, 3)
            if params["unity_camera"]:
                state_0 = np.array(list(env.reset().values())[0][0][0])
                state_1 = np.array(list(env.reset().values())[0][2][0])
                state_feature = agent.check_state(state_0, state_1)
            else:
                state_feature = np.array(list(env.reset().values())[0][0][0])

            episode_reward = 0
            last_actor_loss = 0
            actor_1_losses = []
            actor_2_losses = []
            actor_3_losses = []
            return_list = []
            done = False
            while not done:
                action = agent.take_action(state_feature, actor_index)
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
                    next_state_feature = agent.check_state(next_state_0, next_state_1)

                else:
                    next_state_feature = np.array(list(env.reset().values())[0][0][0])

                reward = np.array(list(reward.values())[0])
                done = np.array(list(done.values())[0])
                agent.store(state_feature, action, reward, next_state_feature, done)

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
                state_feature = next_state_feature

            return_list.append(episode_reward)
            print(
                'episode: {}    actor_1_losses: {}    actor_2_losses: {}  actor_3_losses: {}  episode_reward: '
                '{} '.format(
                    episode, actor_1_losses[-1], actor_2_losses[-1], actor_3_losses[-1], return_list[-1]))

            tags = ["actor_1_losses", "actor_2_losses", "actor_3_losses", "return"]
            tb_writer.add_scalar(tags[0], actor_1_losses[-1], episode)
            tb_writer.add_scalar(tags[1], actor_2_losses[-1], episode)
            tb_writer.add_scalar(tags[2], actor_3_losses[-1], episode)
            tb_writer.add_scalar(tags[3], return_list[-1], episode)

        tb_writer.close()

    else:
        pass
        # agent.load_checkpoints(params["checkpoints_path"])
        # player = Play(env, ma_names, ma_d_action_size, ma_c_action_size, agent, **params)
        # player.agent_eval()
