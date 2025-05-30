import os

import gymnasium as gym

from ...learning.skrl import get_agent
from . import env_cfg

gym.register(
    id="AliengoLeggedEnv-v0",
    entry_point='training.envs.navigation.entrypoints:LeggedEnv',
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.AliengoRoverEnvCfg,
        "best_model_path": f"{os.path.dirname(__file__)}/policies/best_agent.pt",
        "get_agent_fn": get_agent,
    }
)
