import os
import math

from isaaclab.actuators import DCMotorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ManagerTermBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import UniformNoiseCfg

import training.envs.navigation.mdp as user_mdp
from training.envs.navigation.utils.terrains.commands_cfg import TerrainBasedPositionCommandCfg
from training.envs.navigation.utils.terrains.terrain_importer import TerrainBasedPositionCommand
from simulation.agent.agent_ctrl import base_vel_cmd

from .common import EventCfg, RewardsCfg, TerminationsCfg, CurriculumCfg


# todo: modify aliengo configs
UNITREE_Aliengo_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Aliengo",
    spawn=sim_utils.UsdFileCfg(
        usd_path = f"{os.getenv('USER_PATH_TO_USD')}/robots/aliengo/aliengo.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": -0.0,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip": DCMotorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=44.0,
            saturation_effort=44.0,
            velocity_limit=20.0,
            stiffness=40.0,
            damping=2.0,
            friction=0.0,
        ),
        "thigh": DCMotorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=44.0,
            saturation_effort=44.0,
            velocity_limit=20.0,
            stiffness=40.0,
            damping=2.0,
            friction=0.0,
        ),
        "calf": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=55.0,
            saturation_effort=55.0,
            velocity_limit=15.0,
            stiffness=40.0,
            damping=2.0,
            friction=0.0,
        )
    },
)

@configclass
class AliengoSimCfg(InteractiveSceneCfg):
    # ground plane
    # ground = AssetBaseCfg(prim_path="/World/ground",
    #                       spawn=sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300., 300.)),
    #                       init_state=AssetBaseCfg.InitialStateCfg(
    #                           pos=(0, 0, 0.02)
    #                       ))
    
    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),
    )
    cylinder_light = AssetBaseCfg(
        prim_path="/World/cylinderLight",
        spawn=sim_utils.CylinderLightCfg(
            length=100, radius=0.3, treat_as_line=False, intensity=10000.0
        ),
    )
    cylinder_light.init_state.pos = (0, 0, 2.0)

    # Aliengo Robot
    legged_robot: ArticulationCfg = UNITREE_Aliengo_CFG

    # Aliengo foot contact sensor
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Aliengo/.*", history_length=3, track_air_time=True)

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Aliengo/trunk",
        offset=RayCasterCfg.OffsetCfg(pos=[0.0, 0.0, 0.5]),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[5.0, 5.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/Terrain"],
        max_distance=100.0,
    )

@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="legged_robot",
                                           joint_names=[".*"],  # todo: define joint_name seems useless, why?
                                           scale=0.5)

@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel,
        #                        params={"asset_cfg": SceneEntityCfg(name="legged_robot")})

        # Note: himloco policy velocity command is ahead, wmp policy velocity command is behind
        base_vel_cmd = ObsTerm(func=base_vel_cmd, scale=(1, 1, 0.5))  # keyboard 输入的线、角速度分别为 2, 0.5

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25,
                               params={"asset_cfg": SceneEntityCfg(name="legged_robot")})
        projected_gravity = ObsTerm(func=mdp.projected_gravity,
                                    params={"asset_cfg": SceneEntityCfg(name="legged_robot")},
                                    noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05))

        joint_pos = ObsTerm(func=mdp.joint_pos_rel,
                            params={"asset_cfg": SceneEntityCfg(name="legged_robot",)})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05,
                            params={"asset_cfg": SceneEntityCfg(name="legged_robot",)})
        actions = ObsTerm(func=mdp.last_action)

    # @configclass
    # class PlannerPolicyCfg(ObsGroup):
        # todo: use differnet scene env config for locomotion+(navigation) task.
        # comment below obs terms if not with navigation
        distance = ObsTerm(func=user_mdp.distance_to_target_euclidean, params={
            "command_name": "target_pose"}, scale=0.11)
        heading = ObsTerm(func=user_mdp.angle_to_target_observation, params={
                "command_name": "target_pose",}, scale=1/math.pi,
        )
        angle_diff = ObsTerm(func=user_mdp.angle_diff, params={
            "command_name": "target_pose"}, scale=1/math.pi
        )

        height_scan = ObsTerm(func=mdp.height_scan, scale=1,
                              params={"sensor_cfg": SceneEntityCfg("height_scanner"),
                                      # "offset": 0.26878},
                                      "offset": 0.3},   # estimated robot base height
                              clip=(-1.0, 1.0),
                              )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    # planner_policy: PlannerPolicyCfg = PlannerPolicyCfg()

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_vel_cmd = mdp.UniformVelocityCommandCfg(
        asset_name="legged_robot",
        resampling_time_range=(0.0, 0.0),
        heading_command=True,
        rel_heading_envs=1.0,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(-math.pi, math.pi)
        ),
    )

    # need to be comment when not using 'mars' or 'tunnel'.
    # todo: add flat_patch_sampling for different terrains
    target_pose = mdp.TerrainBasedPose2dCommandCfg(
        asset_name="legged_robot",
        resampling_time_range=(50.0, 50.0),
        simple_heading=False,
        debug_vis=True,
        ranges=mdp.TerrainBasedPose2dCommandCfg.Ranges(
            heading=(-math.pi, math.pi)),
    )
    target_pose.goal_pose_visualizer_cfg.markers["arrow"].scale = (1.0, 1.0, 4.0)


@configclass
class AliengoRSLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Aliengo environment."""
    # scene settings
    scene = AliengoSimCfg(num_envs=1, env_spacing=2.0)

    # basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()

    # dummy settings
    commands = CommandsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        # viewer settings
        self.viewer.eye = (-2, 0.0, 0.8)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        # step settings
        self.decimation = 4  # step

        # simulation settings
        self.sim.dt = 0.005  #  0.005  # sim step every
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.render.antialiasing_mode = None    # not supported in lab1.2
        # self.sim.physics_material = self.scene.terrain.physics_material

        # settings for rsl env control
        self.episode_length_s = 20.0 # can be ignored
        self.is_finite_horizon = False
