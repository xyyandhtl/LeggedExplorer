# from typing import Any, Sequence
import torch
from isaaclab.envs.common import VecEnvObs
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporter

from training.envs.navigation.legged_env_cfg import LeggedEnvCfg

VecEnvStepReturn = tuple[VecEnvObs, torch.Tensor,
                         torch.Tensor, torch.Tensor, dict]


class LeggedEnv(ManagerBasedRLEnv):
    """ Legged environment.

    Note:
        This is a placeholder class for the legged environment. That is, this class is not yet implemented."""

    def __init__(self, cfg: LeggedEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        env_ids = torch.arange(self.num_envs, device=self.device)

        # Get the terrain and change the origin
        terrain: TerrainImporter = self.scene.terrain
        terrain.env_origins[env_ids, 0] += 100
        terrain.env_origins[env_ids, 1] += 100

        self.global_step_counter = 0

    #     planner_action_dim = 2
    #     # loco_cmd_dim = 3
    #     self.loco_cmd_vel = torch.zeros((self.num_envs, 3,), device=self.device)
    #     self.planner_last_action = torch.zeros((self.num_envs, 2,), device=self.device)
    #     loco_obs_dim_witout_cmd = 42
    #     loco_obs_dim = 45
    #     planner_obs_dim = (self.observation_manager.group_obs_dim['policy'][0] -
    #                        loco_obs_dim_witout_cmd + planner_action_dim)
    #     self.planner_obs_buf = torch.zeros((self.num_envs, planner_obs_dim,), device=self.device)
    #     self.loco_obs_buf = torch.zeros((self.num_envs, loco_obs_dim,), device=self.device)
    #
    #     self.loco_policy = load_policy_him('aliengo', device=self.device)
    #
    # def reset(
    #     self, seed: int | None = None, env_ids: Sequence[int] | None = None, options: dict[str, Any] | None = None
    # ) -> tuple[VecEnvObs, dict]:
    #
    #     super().reset(seed, env_ids, options)
    #     print(f'after reset obs_buf: {self.obs_buf}')
    #     self._depart_locomotion_planner_obs()
    #
    #     return self.obs_buf, self.extras

    # def _planner_action_to_loco_cmd(self):
    #     # convert planner output action to locomotion input,
    #     self.loco_cmd_vel[:, 0] = self.planner_last_action[:, 0] * 2.0
    #     self.loco_cmd_vel[:, 2] = self.planner_last_action[:, 1] * 0.5
    #     return self.loco_cmd_vel
    #
    # def _depart_locomotion_planner_obs(self):
    #     self._planner_action_to_loco_cmd()
    #     print(f'loco_cmd_vel shape is {self.loco_cmd_vel.shape}')
    #     print(f'obs_buf shape is {self.obs_buf}')
    #     print(f'loco_obs_buf shape is {self.loco_obs_buf.shape}')
    #     self.loco_obs_buf = torch.cat([self.loco_cmd_vel, self.obs_buf['loco_policy']], dim=1)
    #     # self.obs_buf['policy'] = torch.cat([self.planner_last_action, self.obs_buf['policy']], dim=1)
    #     self.obs_buf['policy'][:, :2] = self.planner_last_action

    def _reset_idx(self, idx: torch.Tensor):
        """Reset the environment at the given indices.

        Note:
            This function inherits from :meth:`isaaclab.envs.manager_based_rl_env.ManagerBasedRLEnv._reset_idx`.
            This is done because SKRL requires the "episode" key in the extras dict to be present in order to log.
        Args:
            idx (torch.Tensor): Indices of the environments to reset.
        """
        super()._reset_idx(idx)

        # Done this way because SKRL requires the "episode" key in the extras dict to be present in order to log.
        self.extras["episode"] = self.extras["log"]

    # This function is reimplemented to make visualization less laggy
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        self.global_step_counter += 1

        # self.planner_last_action = action
        # loco_action = self.loco_policy(self.loco_obs_buf)

        # process actions
        # self.action_manager.process_action(loco_action)
        self.action_manager.process_action(action)
        # perform physics stepping
        for _ in range(self.cfg.decimation):
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)
            # perform rendering if gui is enabled
            if self.sim.has_gui():
                self.sim.render()

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()

        # self._depart_locomotion_planner_obs()
        # print(f'output obs shape is {self.obs_buf.shape}')

        assert not torch.isnan(self.obs_buf['policy']).any(), "Observation contains NaN values"
        assert not torch.isinf(self.obs_buf['policy']).any(), "Observation contains Inf values"
        # assert torch.isposinf(self.obs_buf['policy']).any(), "Observation contains posInf values"
        # assert torch.isneginf(self.obs_buf['policy']).any(), "Observation contains negInf values"
        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
