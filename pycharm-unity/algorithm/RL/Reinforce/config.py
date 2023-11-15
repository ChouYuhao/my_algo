import argparse


def get_params():
    parser = argparse.ArgumentParser(
        description="Variable parameters based on the configuration of the machine or user's choice"
    )
    parser.add_argument("--env_name", default=None, type=str, help="Name of the environment")
    # parser.add_argument("--target_update", default=10, type=int, help="The interval steps how often the net update.")
    parser.add_argument("--do_train", default=True, type=bool, help="Start training the agent or playing the agent.")
    parser.add_argument("--mem_size", default=100000, type=int, help="The size of memory of the agent learning.")
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--train_time_scale", default=10, type=int, help="Scale of time to train in Unity.")
    # parser.add_argument("--n_states", default=172, type=int, help="Dims of states in the pyramid environment.")
    parser.add_argument("--n_states", default=2835, type=int, help="Dims of states in the doorKey environment.")

    # parser.add_argument("--n_options", default=3, type=int, help="Dims of options to learn.")
    # parser.add_argument("--n_actions", default=2, type=int, help="Dims of discrete agent actions in Pyramid.")
    parser.add_argument("--n_actions", default=6, type=int, help="Dims of discrete agent actions in doorKey.")
    parser.add_argument("--miniGrid_saveTime", default=100, type=int, help="time of saving model")

    parser.add_argument("--n_features", default=64, type=int, help="")
    parser.add_argument("--if_conv", default=True, type=bool, help="")
    parser.add_argument("--hidden_channels", default=[16, 32, 64], help="")



    parser_param = parser.parse_args()

    default_params = {
        "actor_lr": 1e-4,
        "critic_lr": 1e-3,
        "alpha_lr": 1e-3,
        "tau": 0.001,
        "gamma": 0.98,
        "target_entropy": -1,
        "n_hiddens": 128,
        "batch_size": 256,
        "max_episodes": 100000,
        "max_episode_len": 5000,
    }

    total_params = {**vars(parser_param), **default_params}
    return total_params
