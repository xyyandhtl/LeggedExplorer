from __future__ import annotations

from isaaclab.utils import configclass
import training.envs.navigation.mdp as mdp

from training.envs.navigation.utils.articulation.articulation import AliengoArticulation
from simulation.scene.scene_aliengo import UNITREE_Aliengo_CFG
from training.envs.navigation.legged_env_cfg import LeggedEnvCfg


@configclass
class AliengoRoverEnvCfg(LeggedEnvCfg):
    """Configuration for the AAU rover environment."""

    def __post_init__(self):
        super().__post_init__()

        # Define robot
        self.scene.robot = UNITREE_Aliengo_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",
                                                       class_type=AliengoArticulation)

        # Define parameters for the Ackermann kinematics.
        # self.actions.loco_actions = mdp.JointPositionActionCfg(asset_name="robot",
        #                                    joint_names=[".*"],
        #                                    is_controlled=False,
        #                                    scale=0.5)

        self.actions.actions = mdp.NavigationActionCfg(
            asset_name="robot",
            low_level_decimation=4,
            low_level_action=mdp.JointPositionActionCfg(
                asset_name="robot", joint_names=[".*"], scale=0.5   #, use_default_offset=True
            ),
        )
