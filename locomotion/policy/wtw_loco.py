import torch

from .base import POLICY_WEIGHTS_PATH


def load_policy_wtw(robot_name, device='cuda:0'):
    path1 = f'{POLICY_WEIGHTS_PATH}/wtw_loco/{robot_name}/ada_52507.jit'
    path2 = f'{POLICY_WEIGHTS_PATH}/wtw_loco/{robot_name}/body_52507.jit'
    adap_model = torch.jit.load(path1, map_location=device)
    body_model = torch.jit.load(path2, map_location=device)

    def policy(obs):
        obs_hist = obs.unsqueeze(0)  # (1, N)
        latent = adap_model(obs_hist)
        action = body_model(torch.cat([obs_hist, latent], dim=-1))
        return action[0]

    return policy

