from __future__ import annotations

from typing import TYPE_CHECKING

import torch
# Importing necessary modules from the isaaclab package
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def distance_to_target_reward(env: ManagerBasedRLEnv, command_name: str, max_distance: float) -> torch.Tensor:
    # Accessing the target's position through the command manage,
    # we get the target position w.r.t. the robot frame
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]

    distance = torch.norm(target_position, p=2, dim=-1)
    # training terrain max target distance, try linear reward
    return (max_distance - distance) / env.max_episode_length


def reached_target(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """
    Determine whether the target has been reached.

    This function checks if the rover is within a certain threshold distance from the target.
    If the target is reached, a scaled reward is returned based on the remaining time steps.
    """

    # Accessing the target's position w.r.t. the robot frame
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]

    # Get angle to target
    angle = env.command_manager.get_command(command_name)[:, 3]

    # Calculating the distance and determining if the target is reached
    distance = torch.norm(target_position, p=2, dim=-1)
    time_steps_to_goal = env.max_episode_length - env.episode_length_buf
    reward_scale = time_steps_to_goal / env.max_episode_length

    # Return the reward, scaled depending on the remaining time steps
    return torch.where((distance < threshold) & (torch.abs(angle) < 0.1), 2.0 * reward_scale, 0.0)


def oscillation_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Calculate the oscillation penalty.

    This function penalizes the rover for oscillatory movements by comparing the difference
    in consecutive actions. If the difference exceeds a threshold, a squared penalty is applied.
    """
    # Accessing the rover's actions
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action

    # Calculating differences between consecutive actions
    linear_diff = action[:, 0] - prev_action[:, 0]
    linear_diff2 = action[:, 1] - prev_action[:, 1]
    angular_diff = action[:, 2] - prev_action[:, 2]

    # TODO combine these 5 lines into two lines
    linear_penalty = torch.where(
        linear_diff * 3 > 0.05, torch.square(linear_diff * 3), 0.0)
    linear_penalty2 = torch.where(
        linear_diff2 * 3 > 0.05, torch.square(linear_diff2 * 3), 0.0)
    angular_penalty = torch.where(
        angular_diff*3 > 0.05, torch.square(angular_diff*3), 0.0)

    linear_penalty = torch.pow(linear_penalty, 2)
    linear_penalty2 = torch.pow(linear_penalty2, 2)
    angular_penalty = torch.pow(angular_penalty, 2)

    return (angular_penalty + linear_penalty + linear_penalty2) / env.max_episode_length


def angle_to_target_penalty(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    # Get vector(x,y) from rover to target, in base frame of the rover.
    target_vector_b = env.command_manager.get_command(command_name)[:, :2]

    # Calculate the angle between the rover's heading [1, 0] and the vector to the target.
    angle = torch.atan2(target_vector_b[:, 1], target_vector_b[:, 0])

    # Return the absolute value of the angle, normalized by the maximum episode length.
    return torch.where(torch.abs(angle) > 2.0, torch.abs(angle) / env.max_episode_length, 0.0)


def heading_backwards_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    Calculate a penalty for driving backwards.

    This function applies a penalty when the rover's action indicates reverse movement.
    The penalty is normalized by the maximum episode length.
    """
    return torch.where(env.action_manager.action[:, 0] < 0.0, (1.0 / env.max_episode_length), 0.0)

def lateral_movement_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    base_vel_y = env.command_manager.get_command(command_name)[:, 1]
    return torch.abs(base_vel_y) / env.max_episode_length


def angle_to_goal_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    Calculate the angle to the goal.
    """
    target_vector_b = env.command_manager.get_command(command_name)[:, :2]
    distance = torch.norm(target_vector_b, p=2, dim=-1)
    angle_b = env.command_manager.get_command(command_name)[:, 3]

    mask = distance <= 1.0
    angle_reward = torch.cos(angle_b) / distance * mask
    return angle_reward / env.max_episode_length


def collision_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """
    Calculate a penalty for collisions detected by the sensor.

    This function checks for forces registered by the rover's contact sensor.
    If the total force exceeds a certain threshold, it indicates a collision,
    and a penalty is applied.
    """
    # Accessing the contact sensor and its data
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    force_matrix = contact_sensor.data.force_matrix_w.view(env.num_envs, -1, 3)
    # Calculating the force and applying a penalty if collision forces are detected
    normalized_forces = torch.norm(force_matrix, dim=1)
    forces_active = torch.sum(normalized_forces, dim=-1) > 1
    return torch.where(forces_active, 1.0, 0.0)
