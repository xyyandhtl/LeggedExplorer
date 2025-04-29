import numpy as np
import math

from .base import LocalPlannerIsaac

class LocalPlannerDepth(LocalPlannerIsaac):
    def __init__(self, cfg):
        super().__init__(cfg)
        # self.camera_buf = np.zeros((self.num_envs, 58 * 87 * 9))
        self.obstacle_buf = np.zeros((self.num_envs, 4))
        # self.commands = np.zeros((self.num_envs, 3))  # x vel, y vel, yaw vel, heading

        self.env_last_action = np.zeros((self.num_envs, 3,))
        self.env_left_dis = np.zeros((self.num_envs, 1))
        self.env_right_dis = np.zeros((self.num_envs, 1))

    def infer_action(self):
        if not self.sensor_data['depth']:
            self.commands = np.zeros((self.num_envs, 3))
            return
        assert len(self.sensor_data['depth']) == self.num_envs, \
            f"depth data length {len(self.sensor_data['depth'])} != num_envs {self.num_envs}"
        for env_idx, depth in enumerate(self.sensor_data['depth']):
            if depth is None:
                self.commands[env_idx, :] = 0
                continue
            depth_image = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            # self.camera_buf[env_idx, :] = depth_image.flatten()
            # depth_image = depth_image / 6
            depth_image[depth_image > 1] = 1

            mid_pos = int(0.5 * depth_image.shape[1])
            l_idx, r_idx = -1, -1
            threshold = 0.48
            l_dis, r_dis = threshold, threshold
            l_dis2, r_dis2 = 10, 10
            # min_dis = depth_image[:15, :].max()
            # attention_dis = 2.7
            # scale = 1
            # if self.env_len[env_idx] <= 10:
            #     self.obstacle_buf[env_idx, 0] = depth_image[25:34, :].max()
            self.obstacle_buf[env_idx, 1] = self.obstacle_buf[env_idx, 0]
            self.obstacle_buf[env_idx, 0] = depth_image[25:34, :].max()

            for col in range(2, mid_pos):
                dis = depth_image[:34, mid_pos - col].max()
                if dis < 1.5:
                    if l_dis > dis / math.sqrt(1 + pow(col / mid_pos, 2)) * col / mid_pos:
                        l_dis = dis / math.sqrt(1 + pow(col / mid_pos, 2)) * col / mid_pos
                        l_dis2 = dis
                        l_idx = col

                dis = depth_image[:34, mid_pos + col].max()
                if dis < 1.5:
                    if r_dis > dis / math.sqrt(1 + pow(col / mid_pos, 2)) * col / mid_pos:
                        r_dis = dis / math.sqrt(1 + pow(col / mid_pos, 2)) * col / mid_pos
                        r_dis2 = dis
                        r_idx = col

            if l_dis == threshold or r_dis == threshold:
                for col in range(2, mid_pos):
                    dis = depth_image[:34, mid_pos - col].max()
                    if 1.5 < dis < 3.5:
                        if l_dis > dis / math.sqrt(1 + pow(col / mid_pos, 2)) * col / mid_pos:
                            l_dis = dis / math.sqrt(1 + pow(col / mid_pos, 2)) * col / mid_pos
                            l_dis2 = dis
                            l_idx = col

                    dis = depth_image[:34, mid_pos + col].max()
                    if 1.5 < dis < 3.5:
                        if r_dis > dis / math.sqrt(1 + pow(col / mid_pos, 2)) * col / mid_pos:
                            r_dis = dis / math.sqrt(1 + pow(col / mid_pos, 2)) * col / mid_pos
                            r_dis2 = dis
                            r_idx = col

            if l_dis2 < 1 or r_dis2 < 1:
                if l_dis2 < r_dis2 - 0.15:
                    r_dis = threshold
                elif r_dis2 < l_dis2 - 0.15:
                    l_dis = threshold

            # depth_image = 4 - depth_image
            if l_idx != -1 and r_idx != -1:
                if l_idx <= 3 and r_idx <= 3:
                    # l_num = (depth_image[:20, 0:mid_pos] >= 4).sum().item()
                    # r_num = (depth_image[:20, mid_pos:] >= 4).sum().item()

                    # if (l_num > 20 and r_num < 10) or (l_num > 1.2 * r_num and l_num >= 50):
                    #     l_dis, r_dis = 0.3, 0.15
                    # elif (l_num < 10 and r_num > 20) or (r_num > 1.2 * l_num and r_num >= 50):
                    #     l_dis, r_dis = 0.15, 0.3
                    if 1:
                        down, up = 5, 33
                        # print('*************')
                        # print(depth_image[0:1, :])
                        # #print(depth_image[up-2:up-1,int(mid_pos*1.1):int(mid_pos*1.3)])
                        # #print(depth_image[up-2:up-1,int(mid_pos*0.7):int(mid_pos*0.9)])
                        # print('*************')
                        # print( torch.sum(depth_image[down:up, int(mid_pos*1.1):int(mid_pos*1.3)]), torch.sum(depth_image[down:up, int(mid_pos*0.7):int(mid_pos*0.9)]))
                        if np.sum(depth_image[down:up, int(mid_pos * 1.1):int(mid_pos * 1.3)]) * 1.1 < np.sum(
                                depth_image[down:up, int(mid_pos * 0.7):int(mid_pos * 0.9)]):
                            l_dis, r_dis = 0.3, 0.15
                        elif np.sum(depth_image[down:up, int(mid_pos * 1.1):int(mid_pos * 1.3)]) > 1.1 * np.sum(
                                depth_image[down:up, int(mid_pos * 0.7):int(mid_pos * 0.9)]):
                            l_dis, r_dis = 0.15, 0.3
                        elif np.sum(depth_image[down:up, int(mid_pos * 1.3):int(mid_pos * 1.5)]) * 1.1 < np.sum(
                                depth_image[35:25, int(mid_pos * 0.5):int(mid_pos * 0.8)]):
                            l_dis, r_dis = 0.3, 0.15
                        elif np.sum(depth_image[down:up, int(mid_pos * 1.3):int(mid_pos * 1.5)]) > 1.1 * np.sum(
                                depth_image[down:up, int(mid_pos * 0.5):int(mid_pos * 0.7)]):
                            l_dis, r_dis = 0.15, 0.3
                        elif np.sum(depth_image[down:up, int(mid_pos * 1.8):int(mid_pos * 2)]) * 1.1 < np.sum(
                                depth_image[35:25, 0:int(mid_pos * 0.2)]):
                            l_dis, r_dis = 0.3, 0.15
                        elif np.sum(depth_image[down:up, int(mid_pos * 1.8):int(mid_pos * 2)]) > 1.1 * np.sum(
                                depth_image[down:up, 0:int(mid_pos * 0.2)]):
                            l_dis, r_dis = 0.15, 0.3

            self.obstacle_buf[env_idx, 2] = 0
            self.obstacle_buf[env_idx, 3] = 0

            if l_idx > 3 and r_idx > 3:

                if l_dis == threshold and r_dis < threshold:
                    self.obstacle_buf[env_idx, 3] = 1
                    self.obstacle_buf[env_idx, 2] = 0

                if r_dis == threshold and l_dis < threshold:
                    self.obstacle_buf[env_idx, 3] = 0
                    self.obstacle_buf[env_idx, 2] = 1

            if r_dis == threshold and l_dis == threshold:
                self.commands[env_idx, 0] = 1
                self.commands[env_idx, 1] = 0

                r_scale = 0

                self.commands[env_idx, 2] = r_scale
                # self.commands[env_idx, 3] = r_scale
            else:
                r_scale = -0.1
                forward = 0.85
                if abs(self.env_last_action[env_idx][2]) >= 0.1:
                    forward = 0.75

                if (l_dis > 0.3 and r_dis2 < 2.7 and r_dis < 0.2) or (r_dis > 0.3 and l_dis2 < 2.7 and l_dis < 0.3):
                    r_scale = -0.15

                if (l_dis > 0.3 and r_dis < 0.25 and r_dis2 < 2.8) or (r_dis > 0.3 and l_dis < 0.25 and l_dis2 < 2.8):
                    r_scale = -0.2
                    forward = 0.8

                # if (r_dis <= 0.15 and r_dis2 < 2) or (l_dis <= 0.15 and l_dis2 < 2):
                #     r_scale = -0.25
                #     forward = 0.65

                if (l_dis > 0.3 and r_dis <= 0.15 and r_dis2 < 1.8) or (r_dis > 0.3 and l_dis <= 0.15 and l_dis2 < 1.8):
                    r_scale = -0.25
                    forward = 0.8

                if (r_dis <= 0.2 and r_dis2 < 1 and l_dis >= 0.25) or (l_dis <= 0.2 and l_dis2 < 1 and r_dis >= 0.25):
                    r_scale = -0.3
                    forward = 0.5

                if (r_dis <= 0.15 and r_dis2 < 0.5 and l_dis >= 0.25) or (
                        l_dis <= 0.15 and l_dis2 < 0.5 and r_dis >= 0.25):
                    r_scale = -0.35
                    forward = 0.5

                if r_idx <= 3 and l_idx <= 3 and (l_dis2 < 1.2 or r_dis2 < 1.2) and (l_dis >= 0.3 or r_dis >= 0.3):
                    r_scale = -0.4
                    forward = 0.65

                if (r_dis <= 0.12 and r_dis2 < 0.6 and l_dis == threshold) or (
                        l_dis <= 0.15 and l_dis2 < 0.6 and r_dis == threshold):
                    r_scale = -0.45
                    forward = 0.4

                if l_dis < 0.1 and r_dis < 0.1:
                    forward = min(forward, 0.65)
                # forward = 0.8
                # print(l_dis, l_dis2, r_dis, r_dis2, forward)
                if l_dis < r_dis:
                    # if l_idx <= r_idx:
                    self.commands[env_idx, 0] = forward

                    if max(l_dis2, r_dis2) < 0.5:
                        self.commands[env_idx, 0] = 0.5

                    if l_dis2 > 3:
                        r_scale = -0.05

                    self.commands[env_idx, 1] = 0
                    self.commands[env_idx, 2] = r_scale
                    # self.commands[env_idx, 3] = r_scale
                elif l_dis > r_dis:
                    self.commands[env_idx, 0] = forward

                    if max(l_dis2, r_dis2) < 0.5:
                        self.commands[env_idx, 0] = 0.5

                    if r_dis2 > 3:
                        r_scale = -0.05
                    self.commands[env_idx, 1] = 0
                    self.commands[env_idx, 2] = -r_scale
                    # self.commands[env_idx, 3] = -r_scale

            self.obstacle_buf[env_idx, 0] = 0
            self.obstacle_buf[env_idx, 1] = 0

            if l_dis == threshold and r_dis == threshold:
                self.env_left_dis[env_idx] += 1
                if self.env_left_dis[env_idx] >= 3:
                    self.obstacle_buf[env_idx, 0] = 1
            else:
                self.env_left_dis[env_idx] = 0

            if abs(self.env_last_action[env_idx][2]) >= 0.2:
                self.commands[env_idx, :] = self.commands[env_idx, :] * 0.5 + self.env_last_action[env_idx] * 0.5
            self.env_last_action[env_idx] = self.commands[env_idx, :].clone()
            # print(l_dis, r_dis, l_dis2, r_dis2)




