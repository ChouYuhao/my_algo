import argparse


def get_params():
    parser = argparse.ArgumentParser(
        description="Variable parameters based on the configuration of the machine or user's choice"
    )
    parser.add_argument("--env_name", default=None, type=str, help="Name of the environment")
    parser.add_argument("--target_update", default=10, type=int, help="The interval steps how often the net update.")
    parser.add_argument("--do_train", default=True, type=bool, help="Start training the agent or playing the agent.")
    parser.add_argument("--mem_size", default=20000, type=int, help="The size of memory of the agent learning.")
    parser.add_argument("--seed", default=123, type=int)

    parser.add_argument("--train_time_scale", default=10, type=int, help="Scale of time to train in Unity.")
    parser.add_argument("--n_states", default=172, type=int, help="Dims of states in the environment.")
    parser.add_argument("--n_options", default=3, type=int, help="Dims of options to learn.")
    parser.add_argument("--n_actions", default=5, type=int, help="Dims of discrete agent actions.")

    parser_param = parser.parse_args()

    default_params = {
        "lr": 1e-4,
        "tau": 0.001,
        "gamma": 0.98,
        "batch_size": 256,
        "max_episodes": 100000,
        "max_episode_len": 5000,
    }

    total_params = {**vars(parser_param), **default_params}
    return total_params
