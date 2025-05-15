import torch
# from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from local_planner.policy.base import LocalPlannerBase


class LocalPlannerEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.planner: LocalPlannerBase = None
        self.command_start = self.env.command_start

    def set_local_planner(self, planner: LocalPlannerBase):
        self.planner = planner

    def reset(self):
        self.planner.reset()
        obs, info = self.env.reset()
        if self.env.env_contains_height_scan:
            self.planner.scan_data = self.env.height_scan_buf
        return obs, info

    def step(self, action):
        if self.env.env_contains_height_scan:
            self.planner.scan_data = self.env.height_scan_buf
        cur_cmd_vel = self.planner.get_current_cmd_vel()
        # print(f'planner cur_cmd_vel: {cur_cmd_vel}')
        obs, reward, done, info = self.env.step(action)
        if self.env.obs_history is not None:
            # print(f'obs_history shape {self.env.obs_history.shape}, obs shape {obs.shape}')
            self.env.obs_history[:, self.command_start:self.command_start+3] = torch.from_numpy(cur_cmd_vel).to(obs.device)
        obs[:, self.command_start:self.command_start+3] = torch.from_numpy(cur_cmd_vel).to(obs.device)
        # todo: wtw process
        return obs, reward, done, info
