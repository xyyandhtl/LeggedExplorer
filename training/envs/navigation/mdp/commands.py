"""Sub-module containing command generators for the velocity-based locomotion task."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence
from dataclasses import MISSING
from typing_extensions import Literal

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG


class MidLevelCommandGenerator(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from a path given by a local planner.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    The path follower acts as a PD-controller that checks for the last point on the path within a lookahead distance
    and uses it to compute the steering angle and the linear velocity.
    """

    cfg: MidLevelCommandGeneratorCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: MidLevelCommandGeneratorCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator.

        Args:
            cfg (RLCommandGeneratorCfg): The configuration of the command generator.
            env (object): The environment.
        """
        super().__init__(cfg, env)

        # -- robot
        self.robot: Articulation = env.scene[cfg.robot_attr]
        # -- Simulation Context
        self.sim: SimulationContext = SimulationContext.instance()
        self.command_b: torch.Tensor = torch.zeros((self.num_envs, 3), device=self.device)
        # -- debug vis
        self._base_vel_goal_markers = None
        self._base_vel_markers = None

        # Rotation mark
        self.rotation_mark = False
        self.initialized = False

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "rlCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        # msg += f"\tLookahead distance: {self.cfg.lookAheadDistance}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        # print("twist: ", self.twist)
        return self.command_b

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        """Reset the command generator.

        This function resets the command generator. It should be called whenever the environment is reset.

        Args:
            env_ids (Optional[Sequence[int]], optional): The list of environment IDs to reset. Defaults to None.
        """
        if env_ids is None:
            env_ids = ...

        self.command_b = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_reached = False

        return {}

    def compute(self, dt: float):
        """Compute the command.

        Paths as a tensor of shape (num_envs, N, 3) where N is number of poses on the path. The paths
        should be given in the base frame of the robot. Num_envs is equal to the number of robots spawned in all
        environments.
        """
        # get paths
        self.command_b[:,:2] = self._env.action_manager._terms['actions']._processed_navigation_velocity_actions.clone()[:,:2]

        return self.command_b

    """
    Implementation specific functions.
    """

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


@configclass
class MidLevelCommandGeneratorCfg(CommandTermCfg):
    class_type: MidLevelCommandGenerator = MidLevelCommandGenerator
    """Name of the command generator class."""

    robot_attr: str = MISSING
    """Name of the robot attribute from the environment."""

    path_frame: Literal["world", "robot"] = "world"
    """Frame in which the path is defined.
    - ``world``: the path is defined in the world frame. Also called ``odom``.
    - ``robot``: the path is defined in the robot frame. Also called ``base``.
    """

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""