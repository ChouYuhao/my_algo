import gym
from brain.bc_agent import BC_Agent
# from Common.play import Play
from Common.config import get_params
from Common.play import Play
import numpy as np
from tqdm import tqdm
import torch
import unity_wrapper as uw
from torch.utils.tensorboard import SummaryWriter
import os

if __name__ == "__main__":
    params = get_params()
    torch.manual_seed(123)

    # env = uw.UnityWrapper(train_mode=params["do_preTrain"],
    #                       # file_name=r'E:\Unity\Envs_exe\RLEnvironments.exe',
    #                       # scene='FourRooms',
    #                       n_agents=1,
    #                       no_graphics=False
    #                       )
    # ma_obs_shapes, ma_d_action_size, ma_c_action_size = env.init()
    # ma_names = list(ma_obs_shapes.keys())

    # BC_Agent 传入 demo_file(demo_path)，在 Agent 中进行 demo-> pkl 的转换

    # file = open(params["pkl_filePath"], 'rb')

    agent = BC_Agent(**params)

    if params["do_preTrain"]:
        agent.LoadDataFromUnityDemo()
        tb_writer = SummaryWriter(log_dir="runs/BC_HRL_experiment/WallStop")

        print("Pre_Training is Starting.")

        for episode in range(params["max_n_episodes"]):
            if episode % 1000 == 0:
                agent.save_checkpoints(env_name="UGV", suffix=episode)
            loss = agent.update()
            print('episode: {} loss: {:.4f}'.format(episode + 1, loss.item()))
            tb_writer.add_scalar("loss", loss.item(), episode + 1)
        tb_writer.close()
        # file.close()

    if params["do_prePlay"]:
        # 启动环境
        env = uw.UnityWrapper(train_mode=params["do_prePlay"],
                              # file_name=r'E:\Unity\Envs_exe\RLEnvironments.exe',
                              # scene='FourRooms',
                              n_agents=1,
                              no_graphics=False
                              )
        ma_obs_shapes, ma_d_action_size, ma_c_action_size = env.init()
        ma_names = list(ma_obs_shapes.keys())

        agent.load_checkpoints(params["checkpoints_path"])
        player = Play(env, ma_names, ma_d_action_size, ma_c_action_size, agent, **params)
        player.evaluate_Policy()
