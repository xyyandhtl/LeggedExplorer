import torch

from .base import POLICY_WEIGHTS_PATH


def load_policy_legged(robot_name, device='cuda:0'):
    path = f'{POLICY_WEIGHTS_PATH}/legged_loco/{robot_name}/policy.jit'
    model = torch.jit.load(path, map_location=device)
    model.eval()
    print('Loaded policy from: ', path)

    def policy(obs):
        return model.forward(obs.to(device))
    return policy