from __future__ import annotations

from isaaclab.utils import configclass

import training.mdp as mdp
from training.assets.robots.aau_rover_simple import AAU_ROVER_SIMPLE_CFG
from training.envs.navigation.legged_env_cfg import LeggedEnvCfg


@configclass
class AAURoverEnvCfg(LeggedEnvCfg):
    """Configuration for the AAU rover environment."""

    def __post_init__(self):
        super().__post_init__()

        # Define robot
        self.scene.robot = AAU_ROVER_SIMPLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Define parameters for the Ackermann kinematics.
        self.actions.actions = mdp.AckermannActionCfg(
            asset_name="robot",
            wheelbase_length=0.849,
            middle_wheel_distance=0.894,
            rear_and_front_wheel_distance=0.77,
            wheel_radius=0.1,
            min_steering_radius=0.8,
            steering_joint_names=[".*Steer_Revolute"],
            drive_joint_names=[".*Drive_Continuous"],
            offset=-0.0135
        )
