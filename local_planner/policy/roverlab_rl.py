from pathlib import Path
import numpy as np
import math
import torch
from gymnasium.spaces.box import Box
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory

from training.scripts.config import convert_skrl_cfg
from training.envs.navigation.learning.skrl.models import (Critic, DeterministicActor, DeterministicNeuralNetwork,
                                                           DeterministicNeuralNetworkConv, GaussianNeuralNetwork,
                                                           GaussianNeuralNetworkConv)
from .base import LocalPlannerIsaac


experiment_cfg_agent = {'rollouts': 60, 'learning_epochs': 4, 'mini_batches': 60, 'discount_factor': 0.99, 'lambda': 0.95, 'learning_rate': 0.0001, 'random_timesteps': 0, 'learning_starts': 0, 'grad_norm_clip': 0.5, 'ratio_clip': 0.2, 'value_clip': 0.2, 'clip_predicted_values': True, 'entropy_loss_scale': 0.0, 'value_loss_scale': 1.0, 'kl_threshold': 0.008, 'experiment': {'directory': '/home/lenovo/Opensources/RLRoverLab/examples/03_inference_pretrained/logs/skrl/rover', 'experiment_name': 'May07_09-55-23_PPO', 'write_interval': 40, 'checkpoint_interval': 400, 'wandb': True}}
agent_policy_path = str(Path(__file__).resolve().parent.parent / 'ckpts/roverlab/best_agent.pt')


def get_ppo_agent(device):
    observation_space = Box(low=-math.inf, high=math.inf, shape=(2607,))
    low = np.array([-2.0, -2.0, -0.5], dtype=np.float32)
    high = np.array([2.0, 2.0, 0.5], dtype=np.float32)
    action_space = Box(low=low, high=high, shape=(3,))
    # action_space = Box(low=-1.0, high=1.0, shape=(3,))

    # Define memory size
    memory_size = experiment_cfg_agent["rollouts"]
    memory = RandomMemory(memory_size=memory_size, num_envs=1, device=device)
    # Get the models
    # models = get_models("PPO", env, observation_space, action_space, conv)
    models = {}
    encoder_input_size = 2601

    mlp_input_size = 6

    models["policy"] = GaussianNeuralNetworkConv(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        mlp_input_size=mlp_input_size,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=encoder_input_size,
        encoder_layers=[8, 16, 32, 64],
        encoder_activation="leaky_relu",
    )
    models["value"] = DeterministicNeuralNetworkConv(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        mlp_input_size=mlp_input_size,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=encoder_input_size,
        encoder_layers=[8, 16, 32, 64],
        encoder_activation="leaky_relu",
    )

    # Agent cfg
    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    agent_cfg.update(convert_skrl_cfg(experiment_cfg_agent))

    # Create the agent
    agent = PPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=observation_space,
        action_space=action_space,
        device=device,
    )
    print(f'agent loading from agent_policy_path: {agent_policy_path}')
    agent.load(agent_policy_path)
    agent.set_running_mode("eval")
    return agent  # noqa R504


class LocalPlannerRLRoverLab(LocalPlannerIsaac):
    def __init__(self, cfg):
        super().__init__(cfg)
        # self.camera_buf = np.zeros((self.num_envs, 58 * 87 * 9))
        # self.obstacle_buf = np.zeros((self.num_envs, 4))
        # self.commands = np.zeros((self.num_envs, 3))  # x vel, y vel, yaw vel, heading

        self.env_last_action = torch.zeros((self.num_envs, 3,), device=self.device)
        # self.env_left_dis = np.zeros((self.num_envs, 1))
        # self.env_right_dis = np.zeros((self.num_envs, 1))

        self.agent = get_ppo_agent('cuda:0')
        self.preset_height = 2.0
        self.initialized = False

    def infer_action(self):
        input = torch.cat([self.env_last_action, self.scan_data], dim=1)
        # nan: can't raycast scan dot, regard obstacle
        # posinf: deep holes, regard flat ground considering locomotion can handle
        # neginf: inf high obstacle, theoretically should not contain
        # input = torch.nan_to_num(input, nan=-1.0, posinf=0.0, neginf=-1.0)
        print(f'last_action/distance/heading/diff {input[:, 0:6]}')
        if not self.initialized:
            self.initialized = True
            input_np = input.detach().cpu().numpy()  # 转为 NumPy 数组，便于保存
            # 保存为文本文件
            np.savetxt("input_full.txt", input_np, fmt="%.6f", delimiter=",")
        # print(f'planner input {input}')
        with torch.inference_mode():
            # sample stochastic actions
            # action = self.policy.act(input)[0]
            # with torch.autocast(device_type=self.device, enabled=self.policy._mixed_precision):
            #     actions, log_prob, outputs = self.policy.policy.act(
            #         inputs={"states": self.policy._state_preprocessor(input)},
            #         role="policy")
            #     self.policy._current_log_prob = log_prob
            outputs = self.agent.act(input, timestep=0, timesteps=0)
            # print(f'planner outputs {outputs}')
            # actions = outputs[-1].get("mean_actions", outputs[0])
            actions = outputs[0]    # - 0.0135
            self.env_last_action = actions.clone()
            # actions[:, 0] = actions[:, 0] * 2.0
            # actions[:, 1] = actions[:, 1] * 2.0
            # actions[:, 2] = actions[:, 2] * 0.25
            # print(f'planner actions {actions}')
        # self.commands[:, [0, 2]] = actions.cpu().numpy()
        # self.commands[:, [0, 2]] = actions.cpu().numpy()
        self.commands[:] = actions.cpu().numpy()


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    FILE_PATH = str(Path(__file__).resolve().parent.parent.parent / 'simulation')

    import hydra
    @hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
    def run(cfg):
        planner_test = LocalPlannerRLRoverLab(cfg)
        planner_test.infer_action()

