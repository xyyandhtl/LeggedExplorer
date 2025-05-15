# from typing import Any, Sequence
import numpy as np
import torch
# import omni
# from isaacsim.util.merge_mesh import MeshMerger
from isaaclab.envs.common import VecEnvObs
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
# from isaaclab.terrains import TerrainImporter

from training.envs.navigation.legged_env_cfg import LeggedEnvCfg

VecEnvStepReturn = tuple[VecEnvObs, torch.Tensor,
                         torch.Tensor, torch.Tensor, dict]


class LeggedEnv(ManagerBasedRLEnv):
    """ Legged environment.

    Note:
        This is a placeholder class for the legged environment. That is, this class is not yet implemented."""

    def __init__(self, cfg: LeggedEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # todo: how to merge meshes during setup scene?
        # stage = omni.usd.get_context().get_stage()
        # merger = MeshMerger(stage)
        # merger.output_mesh = "/World/Terrain/Combined"
        # merger.combine_materials = True
        # merger.deactivate_source = False
        # merger.clear_parent_xform = False
        #
        # merger.update_selection([
        #     "/World/Terrain/Ground",
        #     "/World/Terrain/Obstacles",
        # ])
        # merger.merge_meshes()

        # env_ids = torch.arange(self.num_envs, device=self.device)
        # Get the terrain and change the origin
        # terrain: TerrainImporter = self.scene.terrain
        # terrain.env_origins[env_ids, 0] += 100
        # terrain.env_origins[env_ids, 1] += 100
        print(f'env max_episode_length_s is {self.max_episode_length_s}')
        print(f'env max_episode_length is {self.max_episode_length}')

        self.global_step_counter = 0
        self.log_filename = "height_scan_obs.txt"

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

        # process actions
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
            print(f"Resetting envs: {reset_env_ids}")
            self._reset_idx(reset_env_ids)
        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()

        contain_nan = torch.isnan(self.obs_buf['policy']).any()
        contain_inf = torch.isinf(self.obs_buf['policy']).any()
        # 保存计数器
        if not hasattr(self, "_obs_save_counter"):
            self._obs_save_counter = 0
        if self._obs_save_counter % 20 == 0:
            policy_obs = self.obs_buf["policy"].cpu().numpy()
            max_val = policy_obs.max()
            min_val = policy_obs.min()
            with open(self.log_filename, "a") as f:
                # np.savetxt(f, policy_obs, fmt="%.3f")
                f.write(f"Step {self._obs_save_counter}: vel/angular/distance/heading/diff {self.obs_buf['policy'][0, 0:5]}"
                        f"contain_nan={contain_nan}, contain_inf={contain_inf}, "
                        f"max={max_val:.3f}, min={min_val:.3f}\n")
        self._obs_save_counter += 1

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
