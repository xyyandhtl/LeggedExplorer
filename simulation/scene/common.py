import numpy as np
from scipy.spatial.transform import Rotation as R

from omni.isaac.lab.utils import configclass
from omni.isaac.core.utils.viewports import set_camera_view

robot_env_prim_map = {
    'a1': 'A1',
    'go2': 'go2'
}

@configclass
class EventCfg:
    """Configuration for events."""
    pass

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    pass

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass


def camera_follow(env, robot_prim='unitree_a1'):
    if (env.unwrapped.scene.num_envs == 1):
        robot_position = env.unwrapped.scene[robot_prim].data.root_state_w[0, :3].cpu().numpy()
        robot_orientation = env.unwrapped.scene[robot_prim].data.root_state_w[0, 3:7].cpu().numpy()
        rotation = R.from_quat([robot_orientation[1], robot_orientation[2],
                                robot_orientation[3], robot_orientation[0]])
        yaw = rotation.as_euler('zyx')[0]
        yaw_rotation = R.from_euler('z', yaw).as_matrix()
        set_camera_view(
            yaw_rotation.dot(np.asarray([-2, 0.0, 0.4])) + robot_position,
            robot_position
        )

# not-in-use: gym pre-defined env
def get_rsl_env_gym_registered(cfg, robot_name, policy_name):
    import gymnasium as gym
    # cfg.observations.policy.height_scan = None
    env = gym.make(f"Isaac-Velocity-Flat-Unitree-{robot_env_prim_map[robot_name]}-v0", cfg=cfg)
    if policy_name == 'him_loco':
        from locomotion.env_cfg.him_env import HIMLocoEnvWrapper
        env = HIMLocoEnvWrapper(env)
    elif policy_name == 'wmp_loco':
        from locomotion.env_cfg.wmp_env import WMPObsEnvWrapper
        env = WMPObsEnvWrapper(env)
    else:
        raise NotImplementedError
    return env
