import torch
import numpy as np

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from simulation.agent.agent_ctrl import (cmd_base_height, cmd_freq, cmd_stance_length, cmd_stance_width,
                                         cmd_footswing, cmd_pitch, cmd_roll, cmd_gait_type) # all float except gait_type

class WTWLocoEnvWrapper(RslRlVecEnvWrapper):
    def __init__(self, env, history_len=30):
        super().__init__(env)
        self.history_len = history_len
        self.obs_history = None

        self.base_idx = 9
        self.command_start = 6
        # 用于对特定观测段做通道置换
        self.reverse_index_list = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
        # 输出动作的维度反向映射（用于推理部署）
        self.forward_index_list = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]

        self.ABXY2POBD = {
            "A": [0.5, 0.0, 0.0, 0.5],
            "B": [0.0, 0.0, 0.0, 0.5],
            "X": [0.0, 0.5, 0.0, 0.5],
            "Y": [0.0, 0.0, 0.5, 0.5],
        }
        self.policy_dt = env.cfg.sim.dt * env.cfg.decimation
        print('policy_dt: ', self.policy_dt)
        self.gait_idx = 0.0
        self.last_action = None

    def reset(self):
        obs, info = super().reset()
        self.last_action = torch.zeros((obs.shape[0], 12), device=obs.device)
        obs = self._permute_obs(obs)
        # 初始化历史观测，维度：[num_envs, history_len, obs_dim]
        self.obs_history = torch.cat([obs] * self.history_len, dim=-1)
        return self._get_stacked_obs(), info

    def step(self, action):
        action = action.clamp(-5.0, 5.0)
        self.last_action = action   # * 0.25
        action = action[:, self.forward_index_list]

        obs, reward, done, info = super().step(action)
        obs = self._permute_obs(obs)
        # 更新历史观测
        self.obs_history = torch.cat([self.obs_history[:, -obs.shape[1]:], obs.unsqueeze(1)], dim=1)
        return self.obs_history, reward, done, info

    def _get_stacked_obs(self):
        # 展平维度：[num_envs, history_len * obs_dim]
        return self.obs_history.reshape(self.obs_history.shape[0], -1)

    def _permute_obs(self, obs):
        (obs[:, :6], obs[:, 6:9]) = (obs[:, 3:9].clone(), obs[:, :3].clone())
        obs[:, 6] *= 3.5
        obs[:, 7] *= 0.6
        obs[:, 8] *= 5.0
        # joint_pos [base_idx : base_idx+12]
        obs[:, self.base_idx:self.base_idx + 12] = obs[:, self.base_idx:self.base_idx + 12][:, self.reverse_index_list]
        # joint_vel [base_idx+12 : base_idx+24]
        obs[:, self.base_idx + 12:self.base_idx + 24] = obs[:, self.base_idx + 12:self.base_idx + 24][:,
                                                        self.reverse_index_list]
        # last_action [base_idx+24 : base_idx+36]
        obs[:, self.base_idx + 24:self.base_idx + 36] = obs[:, self.base_idx + 24:self.base_idx + 36][:,
                                                        self.reverse_index_list]
        num_envs = obs.shape[0]
        device = obs.device
        cmd_tensor = torch.cat([
            torch.full((num_envs, 1), cmd_base_height(), device=device),
            torch.full((num_envs, 1), cmd_freq(), device=device),
            torch.tensor(self.ABXY2POBD[cmd_gait_type()], device=device).float().unsqueeze(0).repeat(num_envs, 1),
            torch.full((num_envs, 1), cmd_footswing(), device=device),
            torch.full((num_envs, 1), cmd_pitch(), device=device),
            torch.full((num_envs, 1), cmd_roll(), device=device),
            torch.full((num_envs, 1), cmd_stance_width(), device=device),
            torch.full((num_envs, 1), cmd_stance_length(), device=device),
            torch.zeros((num_envs, 1), device=device),  # 用于占位的 0
        ], dim=1)
        obs = torch.cat([obs[:, 3:self.base_idx],
                         cmd_tensor,
                         obs[:, self.base_idx:],
                         self.last_action,
                         torch.tensor(self.get_robot_clock(
                             cmd_freq(),
                             self.ABXY2POBD[cmd_gait_type()][0],
                             self.ABXY2POBD[cmd_gait_type()][1],
                             self.ABXY2POBD[cmd_gait_type()][2]),
                             device=device).float().unsqueeze(0).repeat(num_envs, 1)
                         ], dim=1)
        # print(f'reconstruct obs {obs}')
        return obs

    def get_robot_clock(self, freq, phase, offset, bound):
        self.gait_idx = np.remainder(self.gait_idx + self.policy_dt * freq, 1.0)
        # print('current gait_idx: ', self.gait_idx)
        foot_indices = [
            self.gait_idx + phase + offset + bound,
            self.gait_idx + offset,
            self.gait_idx + bound,
            self.gait_idx + phase
        ]
        return np.sin(2 * np.pi * np.array(foot_indices))
