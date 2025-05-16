from __future__ import annotations
from dataclasses import MISSING

# import time
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from locomotion.policy.him_loco import load_policy_him


# -- Navigation Action
class NavigationAction(ActionTerm):
    """Actions to navigate a robot by following some path."""

    cfg: NavigationActionCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: NavigationActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # self.depth_cnn = DepthImageProcessor(image_height=self.cfg.image_size[0],
        #                                      image_width=self.cfg.image_size[1],num_output_units=32).to(self.device)
        # self.resize_transform = torchvision.transforms.Resize((58, 87),
        #                                                       interpolation=torchvision.transforms.InterpolationMode.BICUBIC)

        # self.image_count = 0

        # # # load policies
        self.low_level_policy = load_policy_him('aliengo', device=self.device)

        # prepare joint position actions
        self.low_level_action_term: ActionTerm = self.cfg.low_level_action.class_type(cfg.low_level_action, env)

        # prepare buffers
        self._action_dim = (
            2
        )  # [vx, vy, omega] --> vx: [-0.5,1.0], vy: [-0.5,0.5], omega: [-1.0,1.0]
        self._raw_navigation_velocity_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_navigation_velocity_actions = torch.zeros(
            (self.num_envs, 3), device=self.device
        )

        self._low_level_actions = torch.zeros(self.num_envs, self.low_level_action_term.action_dim, device=self.device)
        self._low_level_step_dt = self.cfg.low_level_decimation * self._env.physics_dt
        # self._low_level_step_dt = self._env.step_dt
        print(f'low_level_step_dt is {self._low_level_step_dt}')

        self._counter = 0

        self.history_len = 6
        self.obs_history = None

        # self.obs_dim = 45
        self.base_idx = 9
        self.command_start = 0
        # 用于对特定观测段做通道置换
        self.reverse_index_list = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
        # 输出动作的维度反向映射（用于推理部署）
        self.forward_index_list = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
        # self.last_apply_time = time.time()

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_navigation_velocity_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_navigation_velocity_actions

    @property
    def low_level_actions(self) -> torch.Tensor:
        return self._low_level_actions

    """
    Operations.
    """

    def process_actions(self, actions):
        """Process high-level navigation actions. This function is called with a frequency of 10Hz"""

        # Store low level navigation actions
        self._raw_navigation_velocity_actions[:] = actions

        # self._processed_navigation_velocity_actions[:, 0] = actions[:, 0] * 2.0  # todo: not apply scale here
        # self._processed_navigation_velocity_actions[:, 2] = actions[:, 1] * 0.5
        self._processed_navigation_velocity_actions[:, 0] = actions[:, 0]
        self._processed_navigation_velocity_actions[:, 2] = actions[:, 1]

    def apply_actions(self):
        """Apply low-level actions for the simulator to the physics engine. This functions is called with the
        simulation frequency of 200Hz. Since low-level locomotion runs at 50Hz, we need to decimate the actions."""
        # print(f'time from last apply: {time.time() - self.last_apply_time}')
        # self.last_apply_time = time.time()
        if self._counter % self.cfg.low_level_decimation == 0:
            self._counter = 0
            #     # -- update command
            self._env.command_manager.compute(dt=self._low_level_step_dt)
            loco_obs = self._env.observation_manager.compute_group(group_name="low_level_policy")
            loco_obs = self._permute_obs(loco_obs)
            if self.obs_history is None:
                self.obs_history = loco_obs.repeat(1, self.history_len)
            else:
                self.obs_history = torch.cat([loco_obs, self.obs_history[:, :-loco_obs.shape[1]]], dim=1)
            # print(f'loco_obs {loco_obs}')
            # Get low level actions from low level policy
            self._low_level_actions[:] = self.low_level_policy(self.obs_history)[:, self.forward_index_list]
            # print(f'_low_level_actions {self._low_level_actions}')
            # Process low level actions
            self.low_level_action_term.process_actions(self._low_level_actions)

        # Apply low level actions
        self.low_level_action_term.apply_actions()
        self._counter += 1

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


@configclass
class NavigationActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = NavigationAction
    """ Class of the action term."""
    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    low_level_action: ActionTermCfg = MISSING
    """Configuration of the low level action term."""
    # low_level_policy_file: str = MISSING
    """Path to the low level policy file."""
    # path_length: int = 51
    """Length of the path to be followed."""
    # low_level_agent_cfg: dict = {}
    # image_size: tuple = ()
