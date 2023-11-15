import numpy as np


class Play:
    def __init__(self, env, ma_names, ma_d_action_size, ma_c_action_size, agent, **config):
        self.config = config
        self.env = env
        self.ma_names = ma_names
        self.ma_d_action_size = ma_d_action_size
        self.ma_c_action_size = ma_c_action_size
        self.agent = agent
        self.n_skills = self.config["n_skills"]


    @staticmethod
    def concat_state_latent(s, z_, n):
        z_one_hot = np.zeros(n)
        z_one_hot[z_] = 1
        return np.concatenate([s, z_one_hot])

    def agent_eval(self):

        for episode in range(1, self.config["play_n_episode"] + 1):

            if self.config["unity_camera"]:
                state_0 = np.array(list(self.env.reset().values())[0][0][0])
                state_1 = np.array(list(self.env.reset().values())[0][2][0])
                L_state = self.agent.check_state(state_0, state_1)
            else:
                L_state = np.array(list(self.env.reset().values())[0][0][0])
            z = self.agent.take_skill(L_state)
            H_state = self.concat_state_latent(L_state, z, self.n_skills)
            episode_reward = 0
            termination = False

            while True:
                if termination:
                    z = self.agent.take_skill(L_state)
                    H_state = self.concat_state_latent(L_state, z, self.config["n_skills"])
                action = self.agent.take_action(H_state)
                action = action.reshape(1, 2)

                ma_d_action = {}
                ma_c_action = {}
                for n in self.ma_names:
                    d_action, c_action = None, None
                    if self.ma_d_action_size[n]:
                        d_action = action
                        d_action = np.eye(self.ma_d_action_size[n], dtype=np.int32)[d_action]
                    if self.ma_c_action_size[n]:
                        c_action = np.zeros((1, self.ma_c_action_size[n]))
                        c_action = action

                    ma_d_action[n] = d_action
                    ma_c_action[n] = c_action

                next_state, reward, done, _ = self.env.step(ma_d_action, ma_c_action)

                if self.config["unity_camera"]:
                    next_state_0 = np.array(list(next_state.values())[0][0][0])
                    next_state_1 = np.array(list(next_state.values())[0][2][0])
                    next_L_state = self.agent.check_state(next_state_0, next_state_1)
                    next_H_state = self.concat_state_latent(next_L_state, z, self.config["n_skills"])
                else:
                    next_L_state = np.array(list(self.env.reset().values())[0][0][0])
                    next_H_state = self.concat_state_latent(next_L_state, z, self.config["n_skills"])
                reward = np.array(list(reward.values())[0][0])
                done = np.array(list(done.values())[0][0])

                termination = self.agent.get_termination(next_L_state, z)

                episode_reward += reward

                if done:
                    print(f"episode:{episode}, episode_reward:{episode_reward:.1f}")
                    break

                L_state = next_L_state
                H_state = next_H_state



