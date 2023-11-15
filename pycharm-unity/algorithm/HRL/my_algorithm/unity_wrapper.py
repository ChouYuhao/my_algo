import itertools
import logging
import math
import multiprocessing
import multiprocessing.connection
import os
from typing import List

import numpy as np
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel)
from mlagents_envs.side_channel.environment_parameters_channel import \
    EnvironmentParametersChannel

INIT = 0
RESET = 1
STEP = 2
CLOSE = 3
MAX_N_AGENTS_PER_PROCESS = 10


class UnityWrapperProcess:
    def __init__(self,
                 conn: multiprocessing.connection.Connection = None,
                 train_mode=True,
                 file_name=None,
                 worker_id=0,
                 base_port=5005,
                 no_graphics=True,
                 seed=None,
                 scene=None,
                 additional_args=None,
                 n_agents=1):
        """
        Args:
            train_mode: If in train mode, Unity will speed up
            file_name: The executable path. The UnityEnvironment will run in editor if None
            worker_id: Offset from base_port
            base_port: The port that communicate to Unity. It will be set to 5004 automatically if in editor.
            no_graphics: If Unity runs in no graphic mode. It must be set to False if Unity has camera sensor.
            seed: Random seed
            scene: The scene name
            n_agents: The agents count
        """
        self.scene = scene
        self.n_agents = n_agents

        seed = seed if seed is not None else np.random.randint(0, 65536)
        additional_args = [] if additional_args is None else additional_args.split(' ')

        self.engine_configuration_channel = EngineConfigurationChannel()
        self.environment_parameters_channel = EnvironmentParametersChannel()

        self.environment_parameters_channel.set_float_parameter('env_copys', float(n_agents))

        if conn:
            try:
                import config_helper
                config_helper.set_logger()
            except:
                pass

            self._logger = logging.getLogger(f'UnityWrapper.Process_{os.getpid()}')
        else:
            self._logger = logging.getLogger('UnityWrapper.Process')

        self._env = UnityEnvironment(file_name=file_name,
                                     worker_id=worker_id,
                                     base_port=base_port if file_name else None,
                                     no_graphics=no_graphics and train_mode,
                                     seed=seed,
                                     additional_args=['--scene', scene] + additional_args,
                                     side_channels=[self.engine_configuration_channel,
                                                    self.environment_parameters_channel])

        self.engine_configuration_channel.set_configuration_parameters(
            width=200 if train_mode else 1280,
            height=200 if train_mode else 720,
            quality_level=5,
            time_scale=2 if train_mode else 1)

        self._env.reset()
        self.behavior_names = list(self._env.behavior_specs)

        if conn:
            try:
                while True:
                    cmd, data = conn.recv()
                    if cmd == INIT:
                        conn.send(self.init())
                    elif cmd == RESET:
                        conn.send(self.reset(data))
                    elif cmd == STEP:
                        conn.send(self.step(*data))
                    elif cmd == CLOSE:
                        self.close()
            finally:
                self._logger.warning(f'Process {os.getpid()} exits with error')

    def init(self):
        """
        Returns:
            observation shapes: tuple[(o1, ), (o2, ), (o3_1, o3_2, o3_3), ...]
            discrete action size: int, sum of all action branches
            continuous action size: int
        """
        ma_obs_shapes = {}
        self.ma_d_action_size = {}
        self.ma_c_action_size = {}

        for n in self.behavior_names:
            behavior_spec = self._env.behavior_specs[n]
            obs_names = [o.name for o in behavior_spec.observation_specs]
            self._logger.info(f'{n} Observation names: {obs_names}')
            obs_shapes = [o.shape for o in behavior_spec.observation_specs]
            self._logger.info(f'{n} Observation shapes: {obs_shapes}')
            ma_obs_shapes[n] = obs_shapes

            self._empty_action = behavior_spec.action_spec.empty_action

            discrete_action_size = 0
            if behavior_spec.action_spec.discrete_size > 0:
                discrete_action_size = 1
                action_product_list = []
                for action, branch_size in enumerate(behavior_spec.action_spec.discrete_branches):
                    discrete_action_size *= branch_size
                    action_product_list.append(range(branch_size))
                    self._logger.info(f"{n} Discrete action branch {action} has {branch_size} different actions")

                self.action_product = np.array(list(itertools.product(*action_product_list)))

            continuous_action_size = behavior_spec.action_spec.continuous_size

            self._logger.info(f'{n} Continuous action size: {continuous_action_size}')

            self.ma_d_action_size[n] = discrete_action_size
            self.ma_c_action_size[n] = continuous_action_size

            for o in behavior_spec.observation_specs:
                if len(o.shape) >= 3:
                    self.engine_configuration_channel.set_configuration_parameters(quality_level=5)
                    break

        self._logger.info('Initialized')

        return ma_obs_shapes, self.ma_d_action_size, self.ma_c_action_size

    def reset(self, reset_config=None):
        """
        return:
            observations: list[(NAgents, o1), (NAgents, o2), (NAgents, o3_1, o3_2, o3_3)]
        """
        reset_config = {} if reset_config is None else reset_config
        for k, v in reset_config.items():
            self.environment_parameters_channel.set_float_parameter(k, float(v))

        self._env.reset()

        ma_obs_list = {}
        for n in self.behavior_names:
            decision_steps, terminal_steps = self._env.get_steps(n)
            ma_obs_list[n] = [obs.astype(np.float32) for obs in decision_steps.obs]

        return ma_obs_list

    def step(self, ma_d_action, ma_c_action):
        """
        Args:
            d_action: (NAgents, discrete_action_size), one hot like action
            c_action: (NAgents, continuous_action_size)

        Returns:
            observations: list[(NAgents, o1), (NAgents, o2), (NAgents, o3_1, o3_2, o3_3)]
            reward: (NAgents, )
            done: (NAgents, ), bool
            max_step: (NAgents, ), bool
        """
        ma_obs_list = {}
        ma_reward = {}
        ma_done = {}
        ma_max_step = {}

        for n in self.behavior_names:
            d_action = c_action = None

            if self.ma_d_action_size[n]:
                d_action = ma_d_action[n]
                d_action = np.argmax(d_action, axis=1)
                d_action = self.action_product[d_action]
            c_action = ma_c_action[n]

            self._env.set_actions(n,
                                  ActionTuple(continuous=c_action, discrete=d_action))
            self._env.step()

            decision_steps, terminal_steps = self._env.get_steps(n)

            tmp_terminal_steps = terminal_steps

            while len(decision_steps) == 0:
                self._env.set_actions(n, self._empty_action(0))
                self._env.step()
                decision_steps, terminal_steps = self._env.get_steps(n)
                tmp_terminal_steps.agent_id = np.concatenate([tmp_terminal_steps.agent_id,
                                                              terminal_steps.agent_id])
                tmp_terminal_steps.reward = np.concatenate([tmp_terminal_steps.reward,
                                                            terminal_steps.reward])
                tmp_terminal_steps.interrupted = np.concatenate([tmp_terminal_steps.interrupted,
                                                                terminal_steps.interrupted])

            reward = decision_steps.reward
            reward[tmp_terminal_steps.agent_id] = tmp_terminal_steps.reward

            done = np.full([len(decision_steps), ], False, dtype=bool)
            done[tmp_terminal_steps.agent_id] = True

            max_step = np.full([len(decision_steps), ], False, dtype=bool)
            max_step[tmp_terminal_steps.agent_id] = tmp_terminal_steps.interrupted

            ma_obs_list[n] = [obs.astype(np.float32) for obs in decision_steps.obs]
            ma_reward[n] = decision_steps.reward.astype(np.float32)
            ma_done[n] = done
            ma_max_step[n] = max_step

        return ma_obs_list, ma_reward, ma_done, ma_max_step

    def close(self):
        self._env.close()
        self._logger.warning(f'Process {os.getpid()} exits')


class UnityWrapper:
    def __init__(self,
                 train_mode=True,
                 file_name=None,
                 base_port=5005,
                 no_graphics=True,
                 seed=None,
                 scene=None,
                 additional_args=None,
                 n_agents=1,
                 force_seq=None):
        """
        Args:
            train_mode: If in train mode, Unity will run in the highest quality
            file_name: The executable path. The UnityEnvironment will run in editor if None
            base_port: The port that communicate to Unity. It will be set to 5004 automatically if in editor.
            no_graphics: If Unity runs in no graphic mode. It must be set to False if Unity has camera sensor.
            seed: Random seed
            scene: The scene name
            n_agents: The agents count
        """
        self.train_mode = train_mode
        self.file_name = file_name
        self.base_port = base_port
        self.no_graphics = no_graphics
        self.seed = seed
        self.scene = scene
        self.additional_args = additional_args
        self.n_agents = n_agents

        # If use multiple processes
        if force_seq is None:
            self._seq_envs: bool = n_agents <= MAX_N_AGENTS_PER_PROCESS
        else:
            self._seq_envs: bool = force_seq

        self._process_id = 0

        self.env_length = math.ceil(n_agents / MAX_N_AGENTS_PER_PROCESS)

        if self._seq_envs:
            # All environments are executed sequentially
            self._envs: List[UnityWrapperProcess] = []

            for i in range(self.env_length):
                self._envs.append(UnityWrapperProcess(None,
                                                      train_mode,
                                                      file_name,
                                                      i,
                                                      base_port,
                                                      no_graphics,
                                                      seed,
                                                      scene,
                                                      additional_args,
                                                      min(MAX_N_AGENTS_PER_PROCESS, n_agents - i * MAX_N_AGENTS_PER_PROCESS)))
        else:
            # All environments are executed in parallel
            self._conns: List[multiprocessing.connection.Connection] = [None] * self.env_length

            self._generate_processes()

    def _generate_processes(self, force_init=False):
        if self._seq_envs:
            return

        for i, conn in enumerate(self._conns):
            if conn is None:
                parent_conn, child_conn = multiprocessing.Pipe()
                self._conns[i] = parent_conn
                p = multiprocessing.Process(target=UnityWrapperProcess,
                                            args=(child_conn,
                                                  self.train_mode,
                                                  self.file_name,
                                                  self._process_id,
                                                  self.base_port,
                                                  self.no_graphics,
                                                  self.seed,
                                                  self.scene,
                                                  self.additional_args,
                                                  min(MAX_N_AGENTS_PER_PROCESS, self.n_agents - i * MAX_N_AGENTS_PER_PROCESS)),
                                            daemon=True)
                p.start()

                if force_init:
                    parent_conn.send((INIT, None))
                    parent_conn.recv()

                self._process_id += 1

    def init(self):
        """
        Returns:
            observation shapes: dict[str, tuple[(o1, ), (o2, ), (o3_1, o3_2, o3_3), ...]]
            discrete action size: dict[str, int], sum of all action branches
            continuous action size: dict[str, int]
        """
        if self._seq_envs:
            for env in self._envs:
                results = env.init()
        else:
            for conn in self._conns:
                conn.send((INIT, None))
                results = conn.recv()

        self.behavior_names = list(results[0].keys())

        return results

    def reset(self, reset_config=None):
        """
        return:
            observation: dict[str, list[(NAgents, o1), (NAgents, o2), (NAgents, o3_1, o3_2, o3_3)]]
        """
        if self._seq_envs:
            envs_ma_obs_list = [env.reset(reset_config) for env in self._envs]
        else:
            for conn in self._conns:
                conn.send((RESET, reset_config))

            envs_ma_obs_list = [conn.recv() for conn in self._conns]

        ma_obs_list = {n: [np.concatenate(env_obs_list) for env_obs_list in zip(*[ma_obs_list[n] for ma_obs_list in envs_ma_obs_list])]
                       for n in self.behavior_names}

        return ma_obs_list

    def step(self, ma_d_action, ma_c_action):
        """
        Args:
            d_action: dict[str, (NAgents, discrete_action_size)], one hot like action
            c_action: dict[str, (NAgents, continuous_action_size)]

        Returns:
            observation: dict[str, list[(NAgents, o1), (NAgents, o2), (NAgents, o3_1, o3_2, o3_3)]]
            reward: dict[str, (NAgents, )]
            done: dict[str, (NAgents, )], bool
            max_step: dict[str, (NAgents, )], bool
        """
        ma_envs_obs_list = {n: [] for n in self.behavior_names}
        ma_envs_reward = {n: [] for n in self.behavior_names}
        ma_envs_done = {n: [] for n in self.behavior_names}
        ma_envs_max_step = {n: [] for n in self.behavior_names}

        for i in range(self.env_length):
            tmp_ma_d_actions = {
                n:
                ma_d_action[n][i * MAX_N_AGENTS_PER_PROCESS:(i + 1) * MAX_N_AGENTS_PER_PROCESS] if ma_d_action[n] is not None else None
                for n in self.behavior_names}

            tmp_ma_c_actions = {
                n:
                ma_c_action[n][i * MAX_N_AGENTS_PER_PROCESS:(i + 1) * MAX_N_AGENTS_PER_PROCESS] if ma_c_action[n] is not None else None
                for n in self.behavior_names}

            if self._seq_envs:
                (ma_obs_list,
                 ma_reward,
                 ma_done,
                 ma_max_step) = self._envs[i].step(tmp_ma_d_actions, tmp_ma_c_actions)

                for n in self.behavior_names:
                    ma_envs_obs_list[n].append(ma_obs_list[n])
                    ma_envs_reward[n].append(ma_reward[n])
                    ma_envs_done[n].append(ma_done[n])
                    ma_envs_max_step[n].append(ma_max_step[n])
            else:
                self._conns[i].send((STEP, (tmp_ma_d_actions, tmp_ma_c_actions)))

        if not self._seq_envs:
            succeeded = True

            for i, conn in enumerate(self._conns):
                try:
                    (ma_obs_list,
                     ma_reward,
                     ma_done,
                     ma_max_step) = conn.recv()

                    for n in self.behavior_names:
                        ma_envs_obs_list[n].append(ma_obs_list[n])
                        ma_envs_reward[n].append(ma_reward[n])
                        ma_envs_done[n].append(ma_done[n])
                        ma_envs_max_step[n].append(ma_max_step[n])
                except:
                    self._conns[i] = None
                    succeeded = False

            if not succeeded:
                self._generate_processes(force_init=True)

                return None, None, None, None

        ma_obs_list = {n: [np.concatenate(obs) for obs in zip(*ma_envs_obs_list[n])] for n in self.behavior_names}
        ma_reward = {n: np.concatenate(ma_envs_reward[n]) for n in self.behavior_names}
        ma_done = {n: np.concatenate(ma_envs_done[n]) for n in self.behavior_names}
        ma_max_step = {n: np.concatenate(ma_envs_max_step[n]) for n in self.behavior_names}

        return ma_obs_list, ma_reward, ma_done, ma_max_step

    def close(self):
        if self._seq_envs:
            for env in self._envs:
                env.close()
        else:
            for conn in self._conns:
                try:
                    conn.send((CLOSE, None))
                except:
                    pass


