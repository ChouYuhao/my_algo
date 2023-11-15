import argparse



def get_params():
    parser = argparse.ArgumentParser(
        description="Variable parameters based on the configuration of the machine or user's choice")

    parser.add_argument("--env_name", default="Unity", type=str, help="Name of the environment.")
    parser.add_argument("--do_preTrain", default=False, type=bool,
                        help="The flag determines whether to pre_train the agent.")
    parser.add_argument("--do_prePlay", default=True, type=bool,
                        help="The flag determines whether to pre_play the policy of agent.")
    parser.add_argument("--do_train", default=False, type=bool,
                        help="The flag determines whether to pre_train the agent or play it.")
    parser.add_argument("--mem_size", default=20000, type=int, help="The memory size.")
    parser.add_argument("--n_skills", default=20, type=int, help="The number of skills to learn.")
    parser.add_argument("--reward_scale", default=1, type=float, help="The reward scaling factor introduced in SAC.")
    parser.add_argument("--seed", default=123, type=int,
                        help="The randomness' seed for torch, numpy, random & gym[env].")  # 123
    parser.add_argument("--checkpoints_path", default="./checkpoints/BC_HRL/WallStop02/checkpoint_UGV_275000")
    parser.add_argument("--n_states", default=1030, type=int, help="if use unity_camera , value=1024 + 6, else value=6")
    parser.add_argument("--n_actions", default=2, type=int)
    parser.add_argument("--action_bounds", default=[-1, 1])
    parser.add_argument("--unity_camera", default=True, type=bool, help="use camera in unity")
    parser.add_argument("--pkl_filePath", default='E:/Unity/DemoData/BC/traj/WallStop.pkl', type=str, help="pkl path")
    parser.add_argument("--demo_path", default="D:/zyh/705/gitlab/rlenvironments-master/Assets/Demonstrations"
                                               "/WallStop.demo", type=str, help="demo path")
    parser_params = parser.parse_args()

    #  Parameters based on the DIAYN and SAC papers.
    # region default parameters
    default_params = {"lr": 3e-4,
                      "batch_size": 64,
                      "max_n_episodes": 1000000,
                      "max_episode_len": 1000,
                      "gamma": 0.99,
                      "alpha": 0.1,
                      "tau": 0.005,
                      "n_hiddens": 300,
                      }
    # endregion
    total_params = {**vars(parser_params), **default_params}
    return total_params
