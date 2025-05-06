import torch
import numpy as np
from omni.isaac.lab.envs import ManagerBasedEnv

class AgentRecovery:
    def __init__(self, env: ManagerBasedEnv):
        self.env = env
        self.is_fallen = False
        self.recovery_steps = 0
        self.max_recovery_steps = 300  # 增加最大恢复步数
        self.recovery_phase = 0  # 0: 翻转阶段, 1: 站立阶段
        
        # 设置翻转动作的关节位置（弧度）
        self.flip_joint_positions = torch.tensor([
            [1.0, 2.0, -2.0,   # 前左腿 - 最大伸展
             -1.0, 2.0, -2.0,  # 前右腿 - 最大伸展
             1.0, 2.0, -2.0,   # 后左腿 - 最大伸展
             -1.0, 2.0, -2.0], # 后右腿 - 最大伸展
        ], device=env.device)
        
        # 设置站立动作的关节位置（弧度）
        self.stand_joint_positions = torch.tensor([
            [0.1, 0.8, -1.5,   # 前左腿
             -0.1, 0.8, -1.5,  # 前右腿
             0.1, 1.0, -1.5,   # 后左腿
             -0.1, 1.0, -1.5], # 后右腿
        ], device=env.device)
        
    def check_fallen(self, obs):
        """检查机器人是否摔倒"""
        # 获取重力投影向量（第3-6维）
        gravity_proj = obs[:, 3:6]
        
        # 计算重力投影向量的垂直分量（z轴）
        vertical_component = gravity_proj[:, 2]
        
        # 获取关节位置（第9-21维）
        joint_pos = obs[:, 9:21]
        
        # 获取关节速度（第21-33维）
        joint_vel = obs[:, 21:33]
        
        # 判断条件：
        # 1. 重力投影的垂直分量小于阈值（正常站立时接近-1）
        # 2. 关节位置异常（超出正常范围）
        # 3. 关节速度过大
        fallen = (
            (vertical_component > -0.8) |  # 倾斜过大
            (torch.any(torch.abs(joint_pos) > 2.0)) |  # 关节位置异常
            (torch.any(torch.abs(joint_vel) > 5.0))  # 关节速度过大
        )
        
        return fallen
        
    def get_flip_action(self, obs):
        """生成翻转动作"""
        # 从观测中获取当前关节位置
        current_joint_pos = obs[:, 9:21]
        
        # 获取重力投影向量
        gravity_proj = obs[:, 3:6]
        
        # 计算关节位置误差
        joint_error = self.flip_joint_positions - current_joint_pos
        
        # 根据重力投影调整翻转动作
        flip_action = torch.zeros_like(joint_error)
        
        # 如果机器人四脚朝天（重力投影z分量接近1）
        if gravity_proj[0, 2] > 0.5:
            # 尝试用后腿推动翻转
            flip_action[:, 6:9] = torch.tensor([1.0, 1.0, -1.0])  # 左后腿 - 强力伸展
            flip_action[:, 9:12] = torch.tensor([-1.0, 1.0, -1.0])  # 右后腿 - 强力伸展
            # 前腿保持收缩
            flip_action[:, 0:3] = torch.tensor([0.0, -0.5, 0.5])
            flip_action[:, 3:6] = torch.tensor([0.0, -0.5, 0.5])
        # 如果机器人侧翻
        elif abs(gravity_proj[0, 0]) > 0.5:
            # 尝试用对应侧的腿推动翻转
            if gravity_proj[0, 0] > 0:  # 向右倾斜
                flip_action[:, 0:3] = torch.tensor([-1.0, 1.0, -1.0])  # 左前腿
                flip_action[:, 6:9] = torch.tensor([-1.0, 1.0, -1.0])  # 左后腿
            else:  # 向左倾斜
                flip_action[:, 3:6] = torch.tensor([1.0, 1.0, -1.0])  # 右前腿
                flip_action[:, 9:12] = torch.tensor([1.0, 1.0, -1.0])  # 右后腿
        
        # 组合动作，增加翻转动作的权重
        action = joint_error * 0.2 + flip_action * 1.0
        
        return action
        
    def get_stand_action(self, obs):
        """生成站立动作"""
        # 从观测中获取当前关节位置
        current_joint_pos = obs[:, 9:21]
        
        # 计算关节位置误差
        joint_error = self.stand_joint_positions - current_joint_pos
        
        # 使用PD控制器生成动作
        action = joint_error * 0.5
        
        return action
        
    def get_recovery_action(self, obs):
        """根据恢复阶段生成相应的动作"""
        if self.recovery_phase == 0:  # 翻转阶段
            # 检查是否可以进入站立阶段
            gravity_proj = obs[:, 3:6]
            if gravity_proj[0, 2] < -0.5:  # 如果已经基本直立
                print("Switching to standing phase...")
                self.recovery_phase = 1
                return self.get_stand_action(obs)
            return self.get_flip_action(obs)
        else:  # 站立阶段
            return self.get_stand_action(obs)
        
    def update(self, obs, current_action):
        """更新恢复状态并返回适当的动作"""
        if not self.is_fallen:
            # 检查是否摔倒
            fallen = self.check_fallen(obs)
            if fallen.any():
                print("Robot has fallen! Starting recovery...")
                self.is_fallen = True
                self.recovery_steps = 0
                self.recovery_phase = 0  # 从翻转阶段开始
                return self.get_recovery_action(obs)
            return current_action
        else:
            # 执行恢复动作
            self.recovery_steps += 1
            recovery_action = self.get_recovery_action(obs)
            
            # 检查是否已经恢复
            if self.recovery_steps >= self.max_recovery_steps or not self.check_fallen(obs).any():
                print("Recovery completed!")
                self.is_fallen = False
                self.recovery_steps = 0
                self.recovery_phase = 0
                return current_action
                
            return recovery_action 