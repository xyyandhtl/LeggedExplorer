from __future__ import annotations

from typing import TYPE_CHECKING

import torch
# Importing necessary modules from the isaaclab package
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def distance_to_target_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    Calculate and return the distance to the target.

    This function computes the Euclidean distance between the rover and the target.
    It then calculates a reward based on this distance, which is inversely proportional
    to the squared distance. The reward is also normalized by the maximum episode length.
    """

    # Accessing the target's position through the command manage,
    # we get the target position w.r.t. the robot frame
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]

    # Calculating the distance and the reward, closer than 1m clamp to 1
    scaled_distance = torch.clamp(torch.norm(target_position, p=2, dim=-1), 1.0) * 0.1

    # Return the reward, normalized by the maximum episode length
    return 1.0 / scaled_distance / env.max_episode_length


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
    linear_diff = action[:, 1] - prev_action[:, 1]
    angular_diff = action[:, 0] - prev_action[:, 0]

    # TODO combine these 5 lines into two lines
    angular_penalty = torch.where(
        angular_diff*3 > 0.05, torch.square(angular_diff*3), 0.0)
    linear_penalty = torch.where(
        linear_diff*3 > 0.05, torch.square(linear_diff*3), 0.0)

    angular_penalty = torch.pow(angular_penalty, 2)
    linear_penalty = torch.pow(linear_penalty, 2)

    return (angular_penalty + linear_penalty) / env.max_episode_length


def angle_to_target_penalty(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    # Get vector(x,y) from rover to target, in base frame of the rover.
    target_vector_b = env.command_manager.get_command(command_name)[:, :2]

    # Calculate the angle between the rover's heading [1, 0] and the vector to the target.
    angle = torch.atan2(target_vector_b[:, 1], target_vector_b[:, 0])

    # Return the absolute value of the angle, normalized by the maximum episode length.
    return torch.where(torch.abs(angle) > 2.0, torch.abs(angle) / env.max_episode_length, 0.0)


def heading_soft_contraint(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Calculate a penalty for driving backwards.

    This function applies a penalty when the rover's action indicates reverse movement.
    The penalty is normalized by the maximum episode length.
    """
    return torch.where(env.action_manager.action[:, 0] < 0.0, (1.0 / env.max_episode_length), 0.0)


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


def far_from_target_reward(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """
    Gives a penalty if the rover is too far from the target.
    """

    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]

    distance = torch.norm(target_position, p=2, dim=-1)

    return torch.where(distance > threshold, 1.0, 0.0)


def angle_to_goal_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    Calculate the angle to the goal.
    """
    target_vector_b = env.command_manager.get_command(command_name)[:, :2]
    distance = torch.norm(target_vector_b, p=2, dim=-1)
    angle_b = env.command_manager.get_command(command_name)[:, 3]

    # 只在距离 ≤ 1m 时给予奖励
    mask = distance <= 1.0
    # 只对满足条件的样本产生奖励，其余为0
    angle_reward = torch.cos(angle_b) / distance * mask
    return angle_reward / env.max_episode_length


def exploration_reward(env: ManagerBasedRLEnv, command_name: str, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    # Retrieve the robot and target data
    robot: Articulation = env.scene[robot_cfg.name]
    base_velocity = robot.data.root_lin_vel_b  # Robot's current base velocity vector
    target_position = env.command_manager.get_command(command_name)[:, :2]  # Target position vector relative to robot base

    # Compute the dot product of the robot's base velocity and target position vectors
    velocity_alignment = (base_velocity[:, :2] * target_position).sum(-1)

    # Calculate the norms (magnitudes) of the velocity and target position vectors
    velocity_magnitude = torch.norm(base_velocity, p=2, dim=-1)
    target_magnitude = torch.norm(target_position, p=2, dim=-1)

    # Calculate the exploration reward by normalizing the dot product (cosine similarity)
    # Small epsilon added in denominator to prevent division by zero
    exploration_reward = velocity_alignment / (velocity_magnitude * target_magnitude + 1e-6)
    return exploration_reward / env.max_episode_length


def stall_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_vel_threshold: float = 0.1,
    distance_threshold: float = 0.5,
):
    robot: Articulation = env.scene[robot_cfg.name]
    base_vel = robot.data.root_lin_vel_b.norm(2, dim=-1)
    distance_to_goal = env.command_manager.get_command(command_name)[:, :2].norm(2, dim=-1)
    return (base_vel < base_vel_threshold) & (distance_to_goal > distance_threshold)
