import os
import math

from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import training.envs.navigation.mdp as user_mdp
from training.envs.navigation.utils.terrains.commands_cfg import TerrainBasedPositionCommandCfg
from training.envs.navigation.utils.terrains.terrain_importer import TerrainBasedPositionCommand
from simulation.agent.agent_ctrl import base_vel_cmd

from .common import EventCfg, RewardsCfg, TerminationsCfg, CurriculumCfg


GO2W_LEG_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

GO2W_WHEEL_JOINT_NAMES = [
    "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
]

GO2W_JOINT_NAMES = GO2W_LEG_JOINT_NAMES + GO2W_WHEEL_JOINT_NAMES


UNITREE_GO2W_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Go2W",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{os.getenv('USER_PATH_TO_USD')}/robots/go2w/go2w.usd",
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
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=["^(?!.*_foot_joint).*"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit_sim=23.5,
            velocity_limit_sim=30.0,
            stiffness=0.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)


@configclass
class Go2WSimCfg(InteractiveSceneCfg):
    # ground plane
    # ground = AssetBaseCfg(prim_path="/World/ground",
    #                       spawn=sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300., 300.)),
    #                       init_state=AssetBaseCfg.InitialStateCfg(
    #                           pos=(0, 0, 1e-4)
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

    # Go2W Robot
    legged_robot: ArticulationCfg = UNITREE_GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Go2W")

    # Go2W foot contact sensor
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Go2W/.*", history_length=3, track_air_time=True)

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Go2W/base",
        offset=RayCasterCfg.OffsetCfg(pos=[0.0, 0.0, 20.0]),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/Terrain"],
        max_distance=100.0,
    )


@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="legged_robot",
        joint_names=GO2W_LEG_JOINT_NAMES,
        scale={".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25},
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True
    )
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="legged_robot",
        joint_names=GO2W_WHEEL_JOINT_NAMES,
        scale=5.0,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg(name="legged_robot")},
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
       )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg(name="legged_robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        base_vel_cmd = ObsTerm(
            func=base_vel_cmd,
            clip=(-100.0, 100.0),
            scale=(1, 1, 0.5)
        )  # keyboard 输入的线、角速度分别为 2, 0.5
        joint_pos = ObsTerm(
            func=user_mdp.joint_pos_rel_without_wheel,
            params={
                "asset_cfg": SceneEntityCfg(name="legged_robot", joint_names=GO2W_JOINT_NAMES, preserve_order=True),
                "wheel_asset_cfg": SceneEntityCfg(name="legged_robot", joint_names=GO2W_WHEEL_JOINT_NAMES),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="legged_robot", joint_names=GO2W_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0
        )

        # @configclass
        # class PlannerPolicyCfg(ObsGroup):
        # todo: use differnet scene env config for locomotion+(navigation) task.
        # comment below obs terms if not with navigation
        distance = ObsTerm(
            func=user_mdp.distance_to_target_euclidean,
            params={"command_name": "target_pose"}, scale=0.11
        )
        heading = ObsTerm(
            func=user_mdp.angle_to_target_observation,
            params={"command_name": "target_pose", }, scale=1 / math.pi,
        )
        angle_diff = ObsTerm(
            func=user_mdp.angle_diff,
            params={"command_name": "target_pose"}, scale=1 / math.pi
        )

        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        #     scale=1.0,
        # )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ProprioCfg(ObsGroup):
        """Observations for proprioceptive group."""

        # observation terms
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg(name="legged_robot")},
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg(name="legged_robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        base_vel_cmd = ObsTerm(
            func=base_vel_cmd,
            clip=(-100.0, 100.0),
            scale=(1, 1, 0.5)
        )
        joint_pos = ObsTerm(
            func=user_mdp.joint_pos_rel_without_wheel,
            params={
                "asset_cfg": SceneEntityCfg(name="legged_robot", joint_names=GO2W_JOINT_NAMES, preserve_order=True),
                "wheel_asset_cfg": SceneEntityCfg(name="legged_robot", joint_names=GO2W_WHEEL_JOINT_NAMES),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="legged_robot", joint_names=GO2W_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioCfg = ProprioCfg()
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
class EventCfg:
    randomize_reset_base = EventTerm(  # 随机更改env的base 位置、速度（ ± ）
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
                "roll": (2.88, 3.14),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
            "asset_cfg": SceneEntityCfg(name="legged_robot")
        },
    )


@configclass
class Go2WLeggedEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Go2W environment."""
    # scene settings
    scene = Go2WSimCfg(num_envs=1, env_spacing=2.0)

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
        self.sim.dt = 0.005  # 0.005  # sim step every
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.render.antialiasing_mode = None  # not supported in lab1.2
        # self.sim.physics_material = self.scene.terrain.physics_material

        # settings for rsl env control
        self.episode_length_s = 20.0  # can be ignored
        self.is_finite_horizon = False