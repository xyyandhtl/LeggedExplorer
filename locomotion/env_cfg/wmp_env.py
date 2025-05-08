# import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class WMPObsEnvWrapper(RslRlVecEnvWrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.obs_dim = 45
        self.base_idx = 9
        self.command_start = 6
        self.obs_history = None

        # 用于对特定观测段做通道置换
        self.reverse_index_list = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
        # 输出动作的维度反向映射（用于推理部署）
        self.forward_index_list = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]

    def reset(self):
        obs, info = super().reset()
        new_obs = self._construct_padded_obs(obs)
        return new_obs, info

    def step(self, action):
        action = action[:, self.forward_index_list]

        obs, reward, done, info = super().step(action)
        new_obs = self._construct_padded_obs(obs)
        return new_obs, reward, done, info

    def _construct_padded_obs(self, orig_obs):
        # num_envs = orig_obs.shape[0]
        # new_obs = torch.zeros((num_envs, self.wmp_obs_dim_deploy), dtype=orig_obs.dtype, device=orig_obs.device)
        new_obs = orig_obs
        # 做 reverse_index 置换的部分：joint_pos, joint_vel, last_action 各12维
        # wmp cmd_vel is 6:9, while default is 0:3
        (new_obs[:, :6], new_obs[:, 6:9]) = (new_obs[:, 3:9].clone(), new_obs[:, :3].clone())
        return self._permute_obs(new_obs)

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