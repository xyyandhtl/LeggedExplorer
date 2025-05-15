from __future__ import annotations
import math

from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import UniformNoiseCfg
from isaaclab.managers import SceneEntityCfg

import training.envs.navigation.mdp as mdp
from training.envs.navigation.utils.articulation.articulation import AliengoArticulation
from simulation.scene.scene_aliengo import UNITREE_Aliengo_CFG
from training.envs.navigation.legged_env_cfg import LeggedEnvCfg


@configclass
class ObservationsCfg:
    """Observation configuration for the task."""

    @configclass
    class LocoPolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel,
        #                        params={"asset_cfg": SceneEntityCfg(name="robot")})
        # base_velocity = ObsTerm(func=base_vel_cmd)
        base_velocity = ObsTerm(func=mdp.generated_commands, params={"command_name": "midlevel_command"})

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25,
                               params={"asset_cfg": SceneEntityCfg(name="robot")})
        projected_gravity = ObsTerm(func=mdp.projected_gravity,
                                    params={"asset_cfg": SceneEntityCfg(name="robot")},
                                    noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05))

        joint_pos = ObsTerm(func=mdp.joint_pos_rel,
                            params={"asset_cfg": SceneEntityCfg(name="robot",)})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05,
                            params={"asset_cfg": SceneEntityCfg(name="robot",)})
        actions = ObsTerm(func="training.envs.navigation.mdp:low_level_actions") # avoid isaaclab error: cannot pickle 'module' object
        # actions = ObsTerm(func=mdp.low_level_actions)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


    @configclass
    class PlannerPolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        # todo: use differnet scene env config for locomotion+(navigation) task.
        # comment below obs terms if not with navigation
        distance = ObsTerm(func=mdp.distance_to_target_euclidean, params={
            "command_name": "target_pose"}, scale=0.1)
        heading = ObsTerm(func=mdp.angle_to_target_observation, params={
                "command_name": "target_pose",}, scale=1/math.pi,
        )
        angle_diff = ObsTerm(func=mdp.angle_diff, params={
            "command_name": "target_pose"}, scale=1/math.pi
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            scale=1,
            params={"sensor_cfg": SceneEntityCfg(name="height_scanner"),
                    "offset": 0.5 + 0.3},   # estimate robot base height 0.3, add sensor offset 0.5
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PlannerPolicyCfg = PlannerPolicyCfg()
    low_level_policy: LocoPolicyCfg = LocoPolicyCfg()


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    target_pose = mdp.TerrainBasedPose2dCommandCfg(
        class_type=mdp.TerrainBasedPose2dCommand,
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(150.0, 150.0),
        ranges=mdp.TerrainBasedPose2dCommandCfg.Ranges(
            heading=(-math.pi, math.pi)),
        debug_vis=True,
    )
    target_pose.goal_pose_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 2.0)

    midlevel_command: mdp.MidLevelCommandGeneratorCfg = mdp.MidLevelCommandGeneratorCfg(
        robot_attr = "robot",
        debug_vis = True,
        resampling_time_range=(0.0, 0.0),
    )
    midlevel_command.goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 1.0)
    midlevel_command.current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 1.0)


@configclass
class AliengoRoverEnvCfg(LeggedEnvCfg):
    """Configuration for the AAU rover environment."""

    def __post_init__(self):
        # Define robot
        self.scene.robot = UNITREE_Aliengo_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",
                                                       class_type=AliengoArticulation)
        self.observations = ObservationsCfg()
        self.commands = CommandsCfg()
        # Define parameters for the Ackermann kinematics.
        # self.actions.loco_actions = mdp.JointPositionActionCfg(asset_name="robot",
        #                                    joint_names=[".*"],
        #                                    is_controlled=False,
        #                                    scale=0.5)

        self.actions.actions = mdp.NavigationActionCfg(
            asset_name="robot",
            low_level_decimation=4,
            low_level_action=mdp.JointPositionActionCfg(
                asset_name="robot", joint_names=[".*"], scale=0.5  # , use_default_offset=True
            ),
        )

        self.scene.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*(foot|calf|thigh|hip|trunk)",
            # filter_prim_paths_expr=["/World/terrain/obstacles/obstacles"],
            # filter_prim_paths_expr=["/World/Terrain/Obstacles"],
            # todo: /World/Terrain/Obstacles是由HfTunnelTerrainCfg → 生成 heightfield collider；
            # heightfield collider 在 Isaac Sim 中目前不支持 GPU contact filtering（PhysX 限制）；
        )

        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/trunk",
            offset=RayCasterCfg.OffsetCfg(pos=[0.0, 0.0, 0.5]),
            attach_yaw_only=True,
            # pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[5.0, 5.0]),
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[3.0, 3.0]),
            debug_vis=False,    # avoid training fall over markers cannot be zero
            mesh_prim_paths=["/World/Terrain/Obstacles"],
            max_distance=100.0,
        )

        super().__post_init__()
        self.sim.dt = 0.005  # 1 / 30.0 0.005
        self.decimation = 40  # 6
