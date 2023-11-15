import argparse


def get_params():
    parser = argparse.ArgumentParser(
        description="variable parameters in my own algorithm"
    )
    parser.add_argument("--do_train", default=True, type=bool, help="train or not")
    parser.add_argument("--mem_size", default=20000, type=int, help="memory size")
    parser.add_argument("--n_episode", default=100000, type=int, help="the number of episode to train")
    parser.add_argument("--n_steps", default=100, type=int, help="the number of episode to play.")
    parser.add_argument("--reward_scale", default=100, type=float, help="The reward scaling factor introduced in SAC.")
    parser.add_argument("--seed", default=123, type=int,
                        help="The randomness' seed for torch, numpy, random & gym[env].")
    parser.add_argument("--checkpoints_path", default="./checkpoints/FourRooms/MyAlgorithm_checkpoint_UGV_60001")
    parser.add_argument("--n_states", default=32, type=int, help="if use unity_camera value=32, else normal_env value=6")
    parser.add_argument("--n_actions", default=2, type=int)
    parser.add_argument("--action_bound", default=[-1, 1])
    parser.add_argument("--unity_camera", default=True, type=bool, help="if using camera in unity is True, else False")

    parser_params = parser.parse_args()
    default_params = {"lr": 3e-5,
                      "actor_lr": 3e-4,
                      "critic_lr": 3e-3,
                      "batch_size": 32,
                      "sigma": 0.01,
                      "gamma": 0.99,
                      "alpha": 0.1,
                      "tau": 0.005,
                      "n_hidden_filters": 64
                      }

    total_params = {**vars(parser_params), **default_params}
    return total_params
