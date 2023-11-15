import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
# GlfwContext(offscreen=True)


class Play:
    def __init__(self, env, ma_names, ma_d_action_size, ma_c_action_size, agent, n_skills, **config):
        self.config = config
        self.env = env
        self.ma_names = ma_names
        self.ma_d_action_size = ma_d_action_size
        self.ma_c_action_size = ma_c_action_size
        self.agent = agent
        self.n_skills = n_skills
        self.skills_list = []
        self.each_skill_reward = []

    @staticmethod
    def concat_state_latent(s, z_, n):
        z_one_hot = np.zeros(n)
        z_one_hot[z_] = 1
        return np.concatenate([s, z_one_hot])

    def evaluate_skills(self):
        tb_writer = SummaryWriter(log_dir="play/UGV_experiment/skills_3")
        # for z in range(self.n_skills):
        #     # s = self.env.reset()
        #     # s = self.concat_state_latent(s, z, self.n_skills)
        #     s1 = np.array(list(self.env.reset().values())[0][0][0])
        #     s2 = np.array(list(self.env.reset().values())[0][2][0])
        #     s = self.concat_state_latent(self.agent.check_state(s1, s2), z, self.n_skills)
        #     episode_reward = 0
        #     for _ in range(1000):
        #         action = self.agent.choose_action(s)
        #         s_, r, done, _ = self.env.step(action)
        #         s_ = self.concat_state_latent(s_, z, self.n_skills)
        #         episode_reward += r
        #         if done:
        #             break
        #         s = s_
        #     print(f"skill: {z}, episode reward:{episode_reward:.1f}")
        # # self.env.close()

        for z in range(self.n_skills):
            self.skills_list.append(z)

            s1 = np.array(list(self.env.reset().values())[0][0][0])
            s2 = np.array(list(self.env.reset().values())[0][2][0])
            s = self.concat_state_latent(self.agent.check_state(s1, s2), z, self.n_skills)
            episode_reward = 0
            done = False

            for _ in range(20):

                for _ in range(2000):
                    action = self.agent.choose_action(s)
                    action = action.reshape(1, 2)
                    # action[0][0] 是steer
                    # action[0][1] 是motor
                    ma_d_action = {}
                    ma_c_action = {}
                    for n in self.ma_names:
                        d_action, c_action = None, None
                        if self.ma_d_action_size[n]:
                            d_action = action
                            d_action = np.eye(ma_d_action[n], dtype=np.int32)[d_action]
                        if self.ma_c_action_size[n]:
                            c_action = np.zeros((1, self.ma_c_action_size[n]))
                            c_action = action
                        ma_d_action[n] = d_action
                        ma_c_action[n] = c_action

                    s_, r, done, _ = self.env.step(ma_d_action, ma_c_action)
                    s_1 = np.array(list(s_.values())[0][0][0])
                    s_2 = np.array(list(s_.values())[0][2][0])
                    s_ = self.concat_state_latent(self.agent.check_state(s_1, s_2), z, self.n_skills)
                    r = list(r.values())[0][0]
                    done = list(done.values())[0][0]
                    episode_reward += r
                    if done:
                        print(f"skill: {z}, episode reward:{episode_reward:.1f}")
                        tags = ["episode_reward"]

                        # tb_writer.add_scalar(tags[0], z, episode + 1)
                        tb_writer.add_scalar(tags[0], episode_reward, z)
                        break
                    s = s_

        tb_writer.close()

    def evaluate_single_skill(self, skill_index):
        if self.config["unity_camera"]:
            s1 = np.array(list(self.env.reset().values())[0][0][0])
            s2 = np.array(list(self.env.reset().values())[0][2][0])
            s = self.concat_state_latent(self.agent.check_state(s1, s2), skill_index, self.n_skills)
        else:
            s = np.array(list(self.env.reset().values())[0][0][0])
            s = self.concat_state_latent(s, skill_index, self.n_skills)
        episode_reward = 0
        done = False

        for _ in range(200):
            for _ in range(2000):
                action = self.agent.choose_action(s)
                action = action.reshape(1, 2)
                # action[0][0] 是steer
                # action[0][1] 是motor
                ma_d_action = {}
                ma_c_action = {}
                for n in self.ma_names:
                    d_action, c_action = None, None
                    if self.ma_d_action_size[n]:
                        d_action = action
                        d_action = np.eye(ma_d_action[n], dtype=np.int32)[d_action]
                    if self.ma_c_action_size[n]:
                        c_action = np.zeros((1, self.ma_c_action_size[n]))
                        c_action = action
                    ma_d_action[n] = d_action
                    ma_c_action[n] = c_action

                s_, r, done, _ = self.env.step(ma_d_action, ma_c_action)
                if self.config["unity_"]:
                    s_1 = np.array(list(s_.values())[0][0][0])
                    s_2 = np.array(list(s_.values())[0][2][0])
                    s_ = self.concat_state_latent(self.agent.check_state(s_1, s_2), skill_index, self.n_skills)
                else:
                    s_ = np.array(list(s_.values())[0][0][0])
                    s_ = self.concat_state_latent(s, skill_index, self.n_skills)
                r = list(r.values())[0][0]
                done = list(done.values())[0][0]
                episode_reward += r
                if done:
                    print(f"skill: {skill_index}, episode reward:{episode_reward:.1f}")
                    tags = ["episode_reward"]

                    # tb_writer.add_scalar(tags[0], z, episode + 1)
                    # tb_writer.add_scalar(tags[0], episode_reward, z)
                    break
                s = s_
