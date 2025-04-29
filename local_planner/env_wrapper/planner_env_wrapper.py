import torch
# from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper


class LocalPlannerEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.planner = None
        self.command_start = self.env.command_start

    def set_local_planner(self, planner):
        self.planner = planner

    def reset(self):
        self.planner.reset()
        return self.env.reset()

    def step(self, action):
        cur_cmd_vel = self.planner.get_current_cmd_vel()
        print(f'cur_cmd_vel: {cur_cmd_vel}')
        obs, reward, done, info = self.env.step(action)
        if self.env.obs_history is not None:
            print(f'obs_history shape {self.env.obs_history.shape}, obs shape {obs.shape}')
            self.env.obs_history[:, self.command_start:self.command_start+3] = torch.from_numpy(cur_cmd_vel).to(obs.device)
        obs[:, self.command_start:self.command_start+3] = torch.from_numpy(cur_cmd_vel).to(obs.device)
        # todo: wtw process
        return obs, reward, done, info
