import torch

class COTCalculator:
    def __init__(self, robot_data, sim_step_dt):
        self.sim_step_dt = sim_step_dt
        self.mass = robot_data.root_mass[0].item()  # 假设所有 robot 质量一样
        self.num_envs = robot_data.root_pos_w.shape[0]

        # 初始化能量和距离累计
        self.total_energy = torch.zeros(self.num_envs, device=robot_data.root_pos_w.device)
        self.total_distance = torch.zeros(self.num_envs, device=robot_data.root_pos_w.device)

        # 初始位置（用于累计距离）
        self.last_pos = robot_data.root_pos_w.clone()

    def update(self, robot_data):
        joint_torques = robot_data.applied_torque  # [num_envs, num_joints]
        joint_velocities = robot_data.joint_vel    # [num_envs, num_joints]

        # 计算当前步能耗（按绝对功率）
        power = torch.sum(torch.abs(joint_torques * joint_velocities), dim=1)  # [num_envs]
        energy = power * self.sim_step_dt
        self.total_energy += energy

        # 计算前进距离（xy 平面欧式距离）
        current_pos = robot_data.root_pos_w
        delta_xy = current_pos[:, :2] - self.last_pos[:, :2]
        distance = torch.norm(delta_xy, dim=1)
        self.total_distance += distance
        self.last_pos = current_pos.clone()

    def get_cot(self):
        # 避免除以零
        cot = torch.zeros_like(self.total_energy)
        nonzero_mask = self.total_distance > 1e-6
        cot[nonzero_mask] = self.total_energy[nonzero_mask] / (self.mass * 9.81 * self.total_distance[nonzero_mask])
        return cot
