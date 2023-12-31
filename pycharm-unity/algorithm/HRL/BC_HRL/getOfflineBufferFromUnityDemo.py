from mlagents.trainers import demo_loader
import numpy as np
import os
import pickle


# import pandas as pd


def LoadDataFromUnityDemo(demo_path):
    _, info_action_pairs, _ = demo_loader.load_demonstration(demo_path)

    observations = []
    actions = []
    rewards = []
    dones = []
    next_observations = []

    traj_pool = []  # 轨迹库 compressed_data

    state0_pre = np.array(info_action_pairs[0].agent_info.observations[0].float_data.data, dtype=np.float32)
    state1_pre = np.array(info_action_pairs[0].agent_info.observations[1].float_data.data, dtype=np.float32)
    state2_pre = np.array(info_action_pairs[0].agent_info.observations[2].float_data.data, dtype=np.float32)
    state3_pre = np.array(info_action_pairs[0].agent_info.observations[3].float_data.data, dtype=np.float32)

    obs_pre = np.append(state0_pre, state1_pre)
    obs_pre = np.append(obs_pre, state2_pre)
    obs_pre = np.array(np.append(obs_pre, state3_pre))

    # obs_pre = np.array(info_action_pairs[0].agent_info.observations[0].float_data.data, dtype=np.float32)
    done_pre = info_action_pairs[0].agent_info.done

    # print(len(info_action_pairs))
    for info_action_pair in info_action_pairs[1:]:
        agent_info = info_action_pair.agent_info
        action_info = info_action_pair.action_info

        # info_action_pair.action_info.vector_actions_deprecated[0]

        episode_traj = []  # 每个step数据

        state0 = np.array(agent_info.observations[0].float_data.data, dtype=np.float32)
        state1 = np.array(agent_info.observations[1].float_data.data, dtype=np.float32)
        state2 = np.array(agent_info.observations[2].float_data.data, dtype=np.float32)
        state3 = np.array(agent_info.observations[3].float_data.data, dtype=np.float32)

        obs = np.append(state0, state1)
        obs = np.append(obs, state2)
        obs = np.array(np.append(obs, state3))

        rew = agent_info.reward
        # act = np.array(action_info.continuous_actions, dtype=np.float32).vector_actions_deprecated[0]
        act = np.array(action_info.vector_actions_deprecated[0], dtype=np.float32)
        done = agent_info.done
        if not done_pre:
            observations.append(obs_pre)  # 202
            actions.append(act)  # 2
            rewards.append(rew)
            dones.append(done)  #
            next_observations.append(obs)

            s_a = np.append(obs_pre, act)  # 存每个episode
            s_a = np.append(s_a, rew)
            s_a = np.append(s_a, obs)
            s_a = np.append(s_a, done)

            episode_traj.extend([s_a])

        obs_pre = obs
        done_pre = done

        traj_pool.extend(episode_traj)
        del episode_traj[:]

    # pd.DataFrame(traj_pool).to_csv('sample.csv')
    file = open(
        'E:/705(3)/Paper/experience/ml-agents-develop/Project/Assets/ML-Agents/Examples/Pyramids/Demos/pyramid.pkl',
        'wb')
    pickle.dump(traj_pool, file)
    data = dict(
        obs=np.array(observations, dtype=np.float32),
        acts=np.array(actions, dtype=np.float32),
        rews=np.array(rewards, dtype=np.float32),
        next_obs=np.array(next_observations, dtype=np.float32),
        done=np.array(dones, dtype=np.float32),
    )
    return data


def LoadDataFromUnityRosCarDemo2(demo_path):
    """
    状态由图片、雷达、向量组成
    """
    _, info_action_pairs, _ = demo_loader.load_demonstration(demo_path)

    cam_list = []  # 相机
    # rad_list = []  # 雷达
    # vec_list = []  # 向量
    action_list = []
    reward_list = []
    done_list = []

    traj_pool = []  # 轨迹库

    for info_action_pair in info_action_pairs:
        agent_info = info_action_pair.agent_info
        action_info = info_action_pair.action_info

        episode_traj = []  # 每个step数据

        cam = np.array(agent_info.observations[0].float_data.data,
                       dtype=np.float32)  # .reshape((84, 84, 3)).transpose(2,0,1)
        # rad = np.array(agent_info.observations[1].float_data.data, dtype=np.float32)[2:]
        # vec = np.array(agent_info.observations[2].float_data.data, dtype=np.float32)

        cam_list.append(cam)
        # rad_list.append(rad)
        # vec_list.append(vec)
        action_list.append(np.array(action_info.continuous_actions, dtype=np.float32))
        reward_list.append(agent_info.reward)
        done_list.append(agent_info.done)

        s_a = np.append(cam, np.array(action_info.continuous_actions, dtype=np.float32))  # 存每个episode
        # s_ns = np.append(obs_pre,obs)
        episode_traj.extend([s_a])

        traj_pool.extend(episode_traj)
        del episode_traj[:]

    file = open('D:/Desktop/bc/traj/imgcar-2.pkl', 'wb')
    pickle.dump(traj_pool, file)
    data = dict(
        cams=np.array(cam_list, dtype=np.float32),
        # rads=np.array(rad_list, dtype=np.float32),
        # vecs=np.array(vec_list, dtype=np.float32),
        acts=np.array(action_list, dtype=np.float32),
        rews=np.array(reward_list, dtype=np.float32),
        done=np.array(done_list, dtype=np.float32),
    )

    return data


if __name__ == '__main__':
    path = "E:/705(3)/Paper/experience/ml-agents-develop/Project/Assets/ML-Agents/Examples/Pyramids/Demos" \
           "/ExpertPyramid.demo"
    data = LoadDataFromUnityDemo(path)
    print("obs: ", data["obs"].shape)
    print("acts: ", data["acts"].shape)
    print("rews: ", data["rews"].shape)
    print("done: ", data["done"].shape)
