from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab_assets.unitree import UNITREE_A1_CFG
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

@configclass
class A1SimCfg(InteractiveSceneCfg):
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

    # A1 Robot
    unitree_a1: ArticulationCfg = UNITREE_A1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/A1",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.45),  # init height 0.4
            joint_pos={
                ".*L_hip_joint": 0.1,
                ".*R_hip_joint": -0.1,
                "F[L,R]_thigh_joint": 0.8,
                "R[L,R]_thigh_joint": 1.0,
                ".*_calf_joint": -1.5,
            },
            joint_vel={".*": 0.0},
        ),
    )
    unitree_a1.actuators["base_legs"].stiffness = 40.0
    unitree_a1.actuators["base_legs"].damping = 1.0
    print('joint_names_expr:', unitree_a1.actuators["base_legs"].joint_names_expr)

    # A1 foot contact sensor
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/A1/.*_foot", history_length=3, track_air_time=True)

@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="unitree_a1",
                                           joint_names=[".*"],  # todo: define joint_name seems useless, why?
                                           scale=0.25)

@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel,
        #                        params={"asset_cfg": SceneEntityCfg(name="unitree_a1")})

        # Note: himloco policy velocity command is ahead, wmp policy velocity command is behind
        base_vel_cmd = ObsTerm(func=base_vel_cmd)

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25,
                               params={"asset_cfg": SceneEntityCfg(name="unitree_a1")})
        projected_gravity = ObsTerm(func=mdp.projected_gravity,
                                    params={"asset_cfg": SceneEntityCfg(name="unitree_a1")},
                                    noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05))

        joint_pos = ObsTerm(func=mdp.joint_pos_rel,
                            params={"asset_cfg": SceneEntityCfg(name="unitree_a1",)})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05,
                            params={"asset_cfg": SceneEntityCfg(name="unitree_a1",)})
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
        asset_name="unitree_a1",
        resampling_time_range=(0.0, 0.0),
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0, 0)
        ),
    )


@configclass
class A1RSLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the A1 environment."""
    # scene settings
    scene = A1SimCfg(num_envs=1, env_spacing=2.0)

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
