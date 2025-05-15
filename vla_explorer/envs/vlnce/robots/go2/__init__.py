import gymnasium as gym

from .vlnce_mp3d_base_env_cfg import Go2VlnceMp3dBaseEnvCfg, Go2RoughPPORunnerCfg
from .vlnce_mp3d_vision_env_cfg import Go2VlnceMp3dVisionEnvCfg, Go2VisionRoughPPORunnerCfg


gym.register(
    id="Go2Vlnce-MP3D-Base-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2VlnceMp3dBaseEnvCfg,
        "rsl_rl_cfg_entry_point": Go2RoughPPORunnerCfg,
    },
)

gym.register(
    id="Go2Vlnce-MP3D-Base-vision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2VlnceMp3dVisionEnvCfg,
        "rsl_rl_cfg_entry_point": Go2VisionRoughPPORunnerCfg,
    },
)