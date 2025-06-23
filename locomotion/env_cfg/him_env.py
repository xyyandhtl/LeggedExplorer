import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class HIMLocoEnvWrapper(RslRlVecEnvWrapper):
    def __init__(self, env, history_len=6):
        super().__init__(env)
        self.history_len = history_len
        self.obs_history = None

        # self.obs_dim = 45
        self.base_idx = 9
        self.command_start = 0
        # 用于对特定观测段做通道置换 Isaac Lab ==> Isaac Gym
        self.reverse_index_list = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
        # 输出动作的维度反向映射（用于推理部署）Isaac Gym ==> Isaac Lab
        self.forward_index_list = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]

        self.env_contains_height_scan = 'height_scan' in self.env.observation_manager.active_terms['policy']
        if self.env_contains_height_scan:
            idx = self.env.observation_manager.active_terms['policy'].index('height_scan')
            # self.height_scan_dim = self.env.observation_manager.group_obs_term_dim['policy'][idx][0]
            self.height_scan_dim = 2601 + 3
            self.height_scan_buf = torch.zeros((self.env.num_envs, self.height_scan_dim), device=self.env.device)

    def reset(self):
        obs, info = super().reset()
        obs = self._filter_height_scan(obs)
        obs = self._permute_obs(obs)
        # 初始化历史观测，维度：[num_envs, history_len, obs_dim]
        self.obs_history = torch.cat([obs] * self.history_len, dim=1)
        return self.obs_history, info

    def step(self, action):
        action = action[:, self.forward_index_list]

        obs, reward, done, info = super().step(action)
        obs = self._filter_height_scan(obs)
        obs = self._permute_obs(obs)
        # 更新历史观测
        self.obs_history = torch.cat([obs, self.obs_history[:, :-obs.shape[1]]], dim=1)
        return self.obs_history, reward, done, info

    # def _get_stacked_obs(self):
    #     # 展平维度：[num_envs, history_len * obs_dim]
    #     return self.obs_history.reshape(self.obs_history.shape[0], -1)

    def _permute_obs(self, obs):
        # joint_pos [base_idx : base_idx+12]
        obs[:, self.base_idx:self.base_idx + 12] = obs[:, self.base_idx:self.base_idx + 12][:, self.reverse_index_list]
        # joint_vel [base_idx+12 : base_idx+24]
        obs[:, self.base_idx + 12:self.base_idx + 24] = obs[:, self.base_idx + 12:self.base_idx + 24][:,
                                                        self.reverse_index_list]
        # last_action [base_idx+24 : base_idx+36]
        obs[:, self.base_idx + 24:self.base_idx + 36] = obs[:, self.base_idx + 24:self.base_idx + 36][:,
                                                        self.reverse_index_list]
        return obs

    def _filter_height_scan(self, obs):
        if self.env_contains_height_scan:
            self.height_scan_buf = obs[:, -self.height_scan_dim:]
            # print(f'distance/heading/diff {self.height_scan_buf[:, 0:3]}')
            obs = obs[:, :-self.height_scan_dim]
        return obs