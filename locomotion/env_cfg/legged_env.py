import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class LeggedLocoEnvWrapper(RslRlVecEnvWrapper):
    def __init__(self, env, history_length=9):
        super().__init__(env)
        self.history_length = history_length
        self.base_obs_dim = 45
        self.proprio_obs_buf = torch.zeros(self.num_envs, self.history_length, self.base_obs_dim,
                                           dtype=torch.float, device=self.unwrapped.device)

        self.actions_clip = 20

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
        curr_obs = torch.cat([obs] * (self.history_length + 1), dim=1)  # (num_envs, 10 * 45)
        return curr_obs, info

    def step(self, actions):
        actions = torch.clamp(actions, -self.actions_clip, self.actions_clip)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        # (num_envs, 45), (num_envs, 45 + 3 + 187)
        proprio_obs, obs = obs_dict["proprio"], obs_dict["policy"]
        obs = obs[:, :self.base_obs_dim]  # (num_envs, 45)
        extras["observations"] = obs_dict
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # (num_envs, 9, 45)
        self.proprio_obs_buf = torch.where(
            (self.episode_length_buf < 1)[:, None, None],
            torch.stack([torch.zeros_like(proprio_obs)] * self.history_length, dim=1),
            torch.cat([
                self.proprio_obs_buf[:, 1:],
                obs.unsqueeze(1)
            ], dim=1)
        )
        proprio_obs_history = self.proprio_obs_buf.view(self.num_envs, -1)  # (num_envs, 9*45)
        # (num_envs, 45 + 9*45) : 新的观测 + 9个本体观测
        curr_obs = torch.cat([obs, proprio_obs_history], dim=1)
        extras["observations"]["policy"] = curr_obs

        return curr_obs, rew, dones, extras