from __future__ import annotations

import math
import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm  # noqa: F401
from isaaclab.managers import EventTermCfg as EventTerm

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg  # noqa: F401
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import PhysxCfg
from isaaclab.sim import SimulationCfg as SimCfg
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg  # noqa: F401
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg  # noqa: F401

from simulation.terrain.hf_env import TunnelTerrainSceneCfg
import training.envs.navigation.mdp as mdp
# from simulation.terrain.mars_terrains import MarsTerrainSceneCfg
# from training.assets.terrains.debug.debug_terrains import DebugTerrainSceneCfg  # noqa: F401
# from training.assets.terrains.mars import MarsTerrainSceneCfg  # noqa: F401
# from training.envs.navigation.utils.terrains.terrain_importer import RoverTerrainImporter  # noqa: F401
# from training.envs.navigation.utils.terrains.commands_cfg import TerrainBasedPositionCommandCfg
from training.envs.navigation.utils.terrains.terrain_importer import TerrainBasedPositionCommand  # noqa: F401


@configclass
class LeggedSceneCfg(TunnelTerrainSceneCfg):
    """
    Legged Scene Configuration

    Note:
        Terrains can be changed by changing the parent class e.g.
        LeggedSceneCfg(MarsTerrainSceneCfg) -> LeggedSceneCfg(DebugTerrainSceneCfg)

    """
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),
    )
    cylinder_light = AssetBaseCfg(
        prim_path="/World/cylinderLight",
        spawn=sim_utils.CylinderLightCfg(
            length=100, radius=0.3, treat_as_line=False, intensity=5000.0
        ),
    )
    cylinder_light.init_state.pos = (0, 0, 5.0)

    robot: ArticulationCfg = MISSING

    contact_sensor: ContactSensorCfg = MISSING

    height_scanner: RayCasterCfg = MISSING


@configclass
class ActionsCfg:
    """Action"""

    # We define the action space for the legged
    actions: ActionTerm = mdp.NavigationActionCfg(
            asset_name="robot",
            low_level_decimation=4,
            low_level_action=mdp.JointPositionActionCfg(
                asset_name="robot", joint_names=[".*"], scale=0.5  # , use_default_offset=True
            ),
        )


@configclass
class RewardsCfg:
    distance_to_target = RewTerm(
        func=mdp.distance_to_target_reward,
        weight=10.0,
        params={"command_name": "target_pose"},
    )
    reached_target = RewTerm(
        func=mdp.reached_target,
        weight=10.0,
        params={"command_name": "target_pose", "threshold": 0.2},
    )
    oscillation = RewTerm(
        func=mdp.oscillation_penalty,
        weight=-0.1,
        params={},
    )
    angle_to_target = RewTerm(
        func=mdp.angle_to_target_penalty,
        weight=-1.0,
        params={"command_name": "target_pose"},
    )
    heading_soft_contraint = RewTerm(
        func=mdp.heading_soft_contraint,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg(name="robot")},
    )
    # collision = RewTerm(
    #     func=mdp.collision_penalty,
    #     weight=-0.5,    #-3.0,
    #     params={"sensor_cfg": SceneEntityCfg(
    #         "contact_sensor"), "threshold": 1.0},
    # )
    # far_from_target = RewTerm(
    #     func=mdp.far_from_target_reward,
    #     weight=-2.0,
    #     params={"command_name": "target_pose", "threshold": 11.0},
    # )
    angle_diff = RewTerm(
        func=mdp.angle_to_goal_reward,
        weight=50.0,
        params={"command_name": "target_pose"},
    )


@configclass
class TerminationsCfg:
    """Termination conditions for the task."""

    time_limit = DoneTerm(func=mdp.time_out, time_out=True)
    is_success = DoneTerm(
        func=mdp.is_success,
        params={"command_name": "target_pose", "threshold": 0.18},
    )
    far_from_target = DoneTerm(
        func=mdp.far_from_target,
        params={"command_name": "target_pose", "threshold": 50.0},
    )
    # todo: think if add collision DoneTerm?
    # collision = DoneTerm(
    #     func=mdp.collision_with_obstacles,
    #     params={"sensor_cfg": SceneEntityCfg(
    #         "contact_sensor"), "threshold": 1.0},
    # )


@configclass
class EventCfg:
    """Randomization configuration for the task."""
    # startup_state = RandTerm(
    #     func=mdp.reset_root_state_legged,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg(name="robot"),
    #     },
    # )
    reset_state = EventTerm(
        func=mdp.reset_root_state_from_terrain,   # todo: check if valid for legged robot
        mode="reset",
        params={
            "pose_range": {
                # "x": (-1.5, 1.5),
                # "y": (-1.5, 1.5),
                # "z": (0.45, 0.45),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )


# @configclass
# class CurriculumCfg:
#     """ Curriculum configuration for the task. """
#     target_distance = CurrTerm(func=mdp.goal_distance_curriculum)


@configclass
class LeggedEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the legged environment."""

    # Create scene
    scene: LeggedSceneCfg = LeggedSceneCfg(
        num_envs=64, env_spacing=4.0, replicate_physics=False)

    # Setup PhysX Settings
    sim: SimCfg = SimCfg(
        physx=PhysxCfg(
            enable_stabilization=True,
            gpu_max_rigid_contact_count=8388608,
            gpu_max_rigid_patch_count=262144,
            gpu_found_lost_pairs_capacity=2**21,
            gpu_found_lost_aggregate_pairs_capacity=2**25,  # 2**21,
            gpu_total_aggregate_pairs_capacity=2**21,   # 2**13,
            gpu_max_soft_body_contacts=1048576,
            gpu_max_particle_contacts=1048576,
            gpu_heap_capacity=67108864,
            gpu_temp_buffer_capacity=16777216,
            gpu_max_num_partitions=8,
            gpu_collision_stack_size=2**28,
            friction_correlation_distance=0.025,
            friction_offset_threshold=0.04,
            bounce_threshold_velocity=2.0,
        )
    )

    # Basic Settings
    # observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    # MDP Settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # commands: CommandsCfg = CommandsCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.sim.dt = 1 / 30.0   # 1 / 30.0 0.005
        self.decimation = 6     # 6
        self.episode_length_s = 150  # 150 seconds
        # self.viewer.eye = (38, 33, 2)
        # self.viewer.lookat = (33, 33, 0)
        # self.viewer.eye = (-16, 0, 2)
        # self.viewer.lookat = (-10, 0, 0)
        self.viewer.eye = (0, 0, 5)
        self.viewer.lookat = (0, 0, 0)

        # update sensor periods
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        if self.scene.contact_sensor is not None:
            self.scene.contact_sensor.update_period = self.sim.dt * self.decimation


# todo: for flexible use
def create_terrain_based_cfg(terrain_type: str = "mars", scene_id=None):
    env_cfg = LeggedEnvCfg()
    if terrain_type == "mars":
        from simulation.terrain.mars_terrains import hidden_terrain, terrain, obstacles
        env_cfg.scene.terrain = terrain
        env_cfg.scene.hidden_terrain = hidden_terrain
        env_cfg.scene.obstacles = obstacles
        from simulation.terrain.hf_env import tunnel_terrain
        env_cfg.scene.terrain = tunnel_terrain
    elif terrain_type == "obstacle-dense":
        from simulation.terrain.hf_env import dense_obstacle_terrain
        env_cfg.scene.terrain = dense_obstacle_terrain
    elif terrain_type == "omni":
        # todo: remain bugs to fix
        from simulation.terrain.omni_env import omni_terrain_cfg
        env_cfg.scene.terrain = omni_terrain_cfg(scene_id)
    elif terrain_type == "matterport3d":
        from simulation.terrain.mp3d_env import mp3d_terrain_cfg
        env_cfg.scene.terrain = mp3d_terrain_cfg(scene_id)
    elif terrain_type == "carla":
        from simulation.terrain.carla_env import carla_terrain_cfg
        env_cfg.scene.terrain = carla_terrain_cfg()
    else:
        raise NotImplementedError(f'[{terrain_type}] terrain env has not been implemented yet')
    return env_cfg