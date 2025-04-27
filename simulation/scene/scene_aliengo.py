import os

from omni.isaac.lab.actuators import DCMotorCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import RayCasterCfg, patterns, ContactSensorCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.noise import UniformNoiseCfg
from simulation.agent.agent_ctrl import base_vel_cmd

from .common import EventCfg, RewardsCfg, TerminationsCfg, CurriculumCfg


# todo: modify aliengo configs
UNITREE_Aliengo_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Aliengo",
    spawn=sim_utils.UsdFileCfg(
        usd_path = f"{os.getenv('USER_PATH_TO_USD')}/robot/aliengo/aliengo.usd",
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
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=33.5,          # todo
            saturation_effort=33.5,     # todo
            velocity_limit=21.0,        # todo
            stiffness=40.0,
            damping=2.0,
            friction=0.0,
        ),
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
    print('joint_names_expr:', legged_robot.actuators["base_legs"].joint_names_expr)

    # Aliengo foot contact sensor
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Aliengo/.*_foot", history_length=3, track_air_time=True)

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
        base_vel_cmd = ObsTerm(func=base_vel_cmd)

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

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_vel_cmd = mdp.UniformVelocityCommandCfg(
        asset_name="legged_robot",
        resampling_time_range=(0.0, 0.0),
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0, 0)
        ),
    )


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
        self.decimation = 8  # step

        # simulation settings
        self.sim.dt = 0.005  #  0.005  # sim step every
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.render.antialiasing_mode = None    # not supported in lab1.2
        # self.sim.physics_material = self.scene.terrain.physics_material

        # settings for rsl env control
        self.episode_length_s = 20.0 # can be ignored
        self.is_finite_horizon = False
