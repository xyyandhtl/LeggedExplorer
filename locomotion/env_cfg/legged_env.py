import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class LeggedLocoEnvWrapper(RslRlVecEnvWrapper):
    def __init__(self, env, history_length=5):
        super().__init__(env)
        self.history_length = history_length
        self.base_obs_dim = 45
        self.proprio_obs_buf = torch.zeros(self.num_envs, self.history_length, self.base_obs_dim,
                                           dtype=torch.float, device=self.unwrapped.device)

        self.actions_clip = 100
        self.obs_pos_start_idx = 9
        """ dof_names
                Legged-Loco:
                    0 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint'
                    3 'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint'
                    6 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
                    9 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'
                IsaacLab Default:
                    0 'FL_hip_joint',   'FR_hip_joint',   'RL_hip_joint',   'RR_hip_joint',
                    4 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
                    8 'FL_calf_joint',  'FR_calf_joint',  'RL_calf_joint',  'RR_calf_joint',
                IsaacLab Actual (because assign joint_names list in scene_aliengo_lab, and set preserve_order=True. Otherwise, order is default):
                    0 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint'
                    3 'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint'
                    6 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
                    9 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'
                """
        # 用于对特定观测段做通道置换 Isaac Lab ==> Legged-Loco
        # self.reverse_index_list = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]
        self.reverse_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        # 输出动作的维度反向映射（用于推理部署）Legged-Loco ==> Isaac Lab
        # self.forward_index_list = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
        self.forward_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        self.env_contains_height_scan = 'height_scan' in self.env.observation_manager.active_terms['policy']
        if self.env_contains_height_scan:
            idx = self.env.observation_manager.active_terms['policy'].index('height_scan')
            # self.height_scan_dim = self.env.observation_manager.group_obs_term_dim['policy'][idx][0]
            self.height_scan_dim = 187
            self.height_scan_buf = torch.zeros((self.env.num_envs, self.height_scan_dim), device=self.env.device)

    def reset(self):
        # (num_envs, 45 + 3 + 187)
        obs, info = super().reset()
        obs = obs[:, :self.base_obs_dim]  # (num_envs, 45)
        obs = self._permute_obs(obs)
        curr_obs = torch.cat([obs] * (self.history_length + 1), dim=1)  # (num_envs, 6 * 45)
        return curr_obs, info

    def step(self, actions):
        actions = actions[:, self.forward_index_list]
        actions = torch.clamp(actions, -self.actions_clip, self.actions_clip)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        # (num_envs, 45), (num_envs, 45 + 3 + 187)
        proprio_obs, obs = obs_dict["proprio"], obs_dict["policy"]
        obs = obs[:, :self.base_obs_dim]  # (num_envs, 45)

        proprio_obs = self._permute_obs(proprio_obs)
        obs = self._permute_obs(obs)

        extras["observations"] = obs_dict
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # (num_envs, 5, 45)
        self.proprio_obs_buf = torch.where(
            (self.episode_length_buf < 1)[:, None, None],
            torch.stack([torch.zeros_like(proprio_obs)] * self.history_length, dim=1),
            torch.cat([
                proprio_obs.unsqueeze(1),
                self.proprio_obs_buf[:, :-1],
            ], dim=1)
        )
        proprio_obs_history = self.proprio_obs_buf.view(self.num_envs, -1)  # (num_envs, 5*45)
        # (num_envs, 45 + 5*45) : 新的观测 + 5个本体观测
        curr_obs = torch.cat([obs, proprio_obs_history], dim=1)
        extras["observations"]["policy"] = curr_obs

        return curr_obs, rew, dones, extras

    def _permute_obs(self, obs):
        # joint_pos
        obs[:, self.obs_pos_start_idx : self.obs_pos_start_idx + 12] = (
                            obs)[:, self.obs_pos_start_idx:self.obs_pos_start_idx + 12][:, self.reverse_index_list]
        # joint_vel
        obs[:, self.obs_pos_start_idx + 12 : self.obs_pos_start_idx + 24] = (
                            obs)[:, self.obs_pos_start_idx + 12:self.obs_pos_start_idx + 24][:, self.reverse_index_list]
        # last_action
        obs[:, self.obs_pos_start_idx + 24:self.obs_pos_start_idx + 36] = (
                          obs)[:, self.obs_pos_start_idx + 24:self.obs_pos_start_idx + 36][:, self.reverse_index_list]
        return obs


class LeggedLocoGo2WEnvWrapper(LeggedLocoEnvWrapper):
    def __init__(self, env, history_length=5):
        super().__init__(env)
        self.history_length = history_length
        self.base_obs_dim = 57

        self.proprio_obs_buf = torch.zeros(self.num_envs, self.history_length, self.base_obs_dim,
                                           dtype=torch.float, device=self.unwrapped.device)

        """ dof_names
                Legged-Loco:
                    0 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint'
                    3 'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint'
                    6 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
                    9 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'
                    12 'FR_foot_joint', 'FL_foot_joint', 'RR_foot_joint', 'RL_foot_joint'
                IsaacLab Default:
                    0 'FL_hip_joint',   'FR_hip_joint',   'RL_hip_joint',   'RR_hip_joint'
                    4 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
                    8 'FL_calf_joint',  'FR_calf_joint',  'RL_calf_joint',  'RR_calf_joint',
                    12 'FL_foot_joint', 'FR_foot_joint',  'RL_foot_joint',  'RR_foot_joint'
                IsaacLab Actual (because assign joint_names list in scene_go2w_lab, and set preserve_order=True. Otherwise, order is default):
                    0 'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint'
                    3 'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint'
                    6 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
                    9 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'
                    12 'FR_foot_joint', 'FL_foot_joint', 'RR_foot_joint', 'RL_foot_joint'
                """
        # 用于对特定观测段做通道置换 Isaac Lab ==> Legged-Loco
        # self.reverse_index_list = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10, 13, 12, 15, 14]
        self.reverse_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        # 输出动作的维度反向映射（用于推理部署）Legged-Loco ==> Isaac Lab
        # self.forward_index_list = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8, 13, 12, 15, 14]
        self.forward_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        self.env_contains_height_scan = 'height_scan' in self.env.observation_manager.active_terms['policy']
        if self.env_contains_height_scan:
            idx = self.env.observation_manager.active_terms['policy'].index('height_scan')
            # self.height_scan_dim = self.env.observation_manager.group_obs_term_dim['policy'][idx][0]
            self.height_scan_dim = 187
            self.height_scan_buf = torch.zeros((self.env.num_envs, self.height_scan_dim), device=self.env.device)

    def reset(self):
        # (num_envs, 57 + 3(distance+heading+angle_diff) + 187
        obs, info = super().reset()  # info: {'observations': {'policy': tensor(num_envs, 57+3)}}
        obs = obs[:, :self.base_obs_dim]  # (num_envs, 57)
        obs = self._permute_obs(obs)  # ==> policy
        curr_obs = torch.cat([obs] * (self.history_length + 1), dim=1)  # (num_envs, 6 * 57)
        return curr_obs, info

    def step(self, actions):
        actions = actions[:, self.forward_index_list]  # ==> isaaclab
        actions = torch.clamp(actions, -self.actions_clip, self.actions_clip)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        # (num_envs, 57), (57, 45 + 3 + 187)
        proprio_obs, obs = obs_dict["proprio"], obs_dict["policy"]
        obs = obs[:, :self.base_obs_dim]  # (num_envs, 57)

        proprio_obs = self._permute_obs(proprio_obs)  # ==> policy
        obs = self._permute_obs(obs)

        extras["observations"] = obs_dict
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # (num_envs, 5, 57)
        self.proprio_obs_buf = torch.where(
            (self.episode_length_buf < 1)[:, None, None],
            torch.stack([torch.zeros_like(proprio_obs)] * self.history_length, dim=1),
            torch.cat([
                proprio_obs.unsqueeze(1),
                self.proprio_obs_buf[:, :-1],
            ], dim=1)
        )
        proprio_obs_history = self.proprio_obs_buf.view(self.num_envs, -1)  # (num_envs, 5*57)
        # (num_envs, 57 + 5*57) : 新的观测 + 5个本体观测
        curr_obs = torch.cat([obs, proprio_obs_history], dim=1)
        extras["observations"]["policy"] = curr_obs

        return curr_obs, rew, dones, extras

    def _permute_obs(self, obs):
        # joint_pos
        obs[:, self.obs_pos_start_idx : self.obs_pos_start_idx + 16] = (
                            obs)[:, self.obs_pos_start_idx : self.obs_pos_start_idx + 16][:, self.reverse_index_list]
        # joint_vel
        obs[:, self.obs_pos_start_idx + 16 : self.obs_pos_start_idx + 32] = (
                            obs)[:, self.obs_pos_start_idx + 16 : self.obs_pos_start_idx + 32][:, self.reverse_index_list]
        # last_action
        obs[:, self.obs_pos_start_idx + 32 : self.obs_pos_start_idx + 48] = (
                          obs)[:, self.obs_pos_start_idx + 32 : self.obs_pos_start_idx + 48][:, self.reverse_index_list]
        return obs