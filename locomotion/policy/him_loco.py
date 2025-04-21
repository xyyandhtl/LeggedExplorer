import torch

from .base import POLICY_WEIGHTS_PATH


def load_policy_him(robot_name, device='cuda:0'):
    path = f'{POLICY_WEIGHTS_PATH}/him_loco/{robot_name}/policy.pt'
    body = torch.jit.load(path, map_location=device)
    print('Loaded policy from: ', path)
    def policy(obs):
        return body.forward(obs.to(device))
    return policy