from __future__ import annotations
import math

from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import UniformNoiseCfg
from isaaclab.managers import SceneEntityCfg

# from training.envs.navigation.utils.terrains.terrain_importer import TerrainBasedPositionCommand
# from training.envs.navigation.utils.terrains.commands_cfg import TerrainBasedPositionCommandCfg
import training.envs.navigation.mdp as mdp
from training.assets.robots.aau_rover_simple import AAU_ROVER_SIMPLE_CFG
from training.envs.navigation.legged_env_cfg import LeggedEnvCfg


@configclass
class ObservationsCfg:
    """Observation configuration for the task."""

    @configclass
    class PlannerPolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        # todo: use differnet scene env config for locomotion+(navigation) task.
        # comment below obs terms if not with navigation
        distance = ObsTerm(func=mdp.distance_to_target_euclidean, params={
            "command_name": "target_pose"}, scale=0.11)
        heading = ObsTerm(func=mdp.angle_to_target_observation, params={
                "command_name": "target_pose",}, scale=1/math.pi,
        )
        angle_diff = ObsTerm(func=mdp.angle_diff, params={
            "command_name": "target_pose"}, scale=1/math.pi
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            scale=1,
            params={"sensor_cfg": SceneEntityCfg(name="height_scanner"), "offset": 0.26878},
            # noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PlannerPolicyCfg = PlannerPolicyCfg()


# "mdp.illegal_contact
@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    target_pose = mdp.TerrainBasedPose2dCommandCfg(
        class_type=mdp.TerrainBasedPose2dCommand,
        asset_name="robot",
        # rel_standing_envs=0.0,
        simple_heading=False,
        resampling_time_range=(150.0, 150.0),
        ranges=mdp.TerrainBasedPose2dCommandCfg.Ranges(
            heading=(-math.pi, math.pi)),
        debug_vis=True,
    )
    target_pose.goal_pose_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 2.0)


@configclass
class AAURoverEnvCfg(LeggedEnvCfg):
    """Configuration for the AAU rover environment."""

    def __post_init__(self):
        # Define robot
        self.scene.robot = AAU_ROVER_SIMPLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.observations = ObservationsCfg()
        self.commands = CommandsCfg()

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

        # self.scene.contact_sensor = ContactSensorCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/.*_(Drive|Steer|Boogie|Body)",
        #     # filter_prim_paths_expr=["/World/terrain/obstacles/obstacles"],
        #     # filter_prim_paths_expr=["/World/Terrain/Obstacles"],
        # )

        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Body",
            offset=RayCasterCfg.OffsetCfg(pos=[0.0, 0.0, 10.0]),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[5.0, 5.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/Terrain/Obstacles"],
            max_distance=100.0,
        )

        super().__post_init__()
        self.sim.dt = 1 / 30.0  # 1 / 30.0 0.005
        self.decimation = 6  # 6
