from __future__ import annotations

from typing import TYPE_CHECKING

import torch
# from isaaclab.managers import SceneEntityCfg
# from isaaclab.sensors import RayCaster

from .actions import NavigationAction

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
# from isaaclab.command_generators import UniformPoseCommandGenerator

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def angle_to_target_observation(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Calculate the angle to the target."""

    # Get vector(x,y) from rover to target, in base frame of the rover.
    target_vector_b = env.command_manager.get_command(command_name)[:, :2]

    # Calculate the angle between the rover's heading [1, 0] and the vector to the target.
    angle = torch.atan2(target_vector_b[:, 1], target_vector_b[:, 0])

    return angle.unsqueeze(-1)


def distance_to_target_euclidean(env: ManagerBasedRLEnv, command_name: str):
    """Calculate the euclidean distance to the target."""
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]
    distance: torch.Tensor = torch.norm(target_position, p=2, dim=-1)
    return distance.unsqueeze(-1)


# def height_scan_legged(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Calculate the height scan of the rover.
#
#     This function uses a ray caster to generate a height scan of the rover's surroundings.
#     The height scan is normalized by the maximum range of the ray caster.
#     """
#     # extract the used quantities (to enable type-hinting)
#     sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
#     # height scan: height = sensor_height - hit_point_z - 0.26878
#     # Note: 0.26878 is the distance between the sensor and the rover's base
#     return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - 0.26878


def angle_diff(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Calculate the angle difference between the rover's heading and the target."""
    # Get the angle to the target
    heading_angle_diff = env.command_manager.get_command(command_name)[:, 3]

    return heading_angle_diff.unsqueeze(-1)


def low_level_actions(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Low-level actions."""
    # extract the used quantities (to enable type-hinting)
    # action_term: NavigationAction = env.action_manager._terms['actions']
    action_term: NavigationAction = env.action_manager.get_term('actions')

    return action_term.low_level_actions.clone()


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel