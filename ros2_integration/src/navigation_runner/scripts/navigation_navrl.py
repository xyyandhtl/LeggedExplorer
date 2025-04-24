#!/usr/bin/env python3
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped, Quaternion, Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool
from builtin_interfaces.msg import Duration
from map_manager.srv import RayCast
from onboard_detector.srv import GetDynamicObstacles
from navigation_runner.srv import GetSafeAction
import torch
import numpy as np
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from tensordict.tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from navrl.ppo import PPO
from navrl.utils import vec_to_new_frame, vec_to_world
from pid_controller import AnglePIDController
import os


class Navigation(Node):
    def __init__(self, cfg):
        super().__init__('navigation_node')
        self.cfg = cfg
        self.lidar_hbeams = int(360/self.cfg.sensor.lidar_hres)
        self.raypoints = []
        self.dynamic_obstacles = []
        self.robot_size = 0.3 # radius
        self.raycast_vres = ((self.cfg.sensor.lidar_vfov[1] - self.cfg.sensor.lidar_vfov[0]))/(self.cfg.sensor.lidar_vbeams - 1) * np.pi/180.0
        self.raycast_hres = self.cfg.sensor.lidar_hres * np.pi/180.0

        self.goal = None
        self.goal_received = False
        self.target_dir = None
        self.stable_times = 0
        self.has_action = False
        self.laser_points_msg = None

        self.height_control = False 
        self.use_policy_server = False
        self.odom_received = False
        self.safety_stop = False
    
        # Velocity limit
        self.declare_parameter('vel_limit', 1.0)
        self.vel_limit = self.get_parameter('vel_limit').get_parameter_value().double_value
        self.get_logger().info(f"[navRunner]: Velocity limit: {self.vel_limit}.")

        # Visualize raycast
        self.declare_parameter('visualize_raycast', False)
        self.vis_raycast = self.get_parameter('visualize_raycast').get_parameter_value().bool_value
        self.get_logger().info(f"[navRunner]: Visualize raycast is set to: {self.vis_raycast}.")

        # Subscriber
        self.declare_parameter('odom_topic', '/legged/odom')
        odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.get_logger().info(f"[navRunner]: Odom topic name: {odom_topic}.")
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10) # odom
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10) # goal
        self.emergency_stop_sub = self.create_subscription(Bool, '/navigation_emergency_stop', self.safety_check_callback, 10) # safety check
        
        # Publisher
        self.declare_parameter('cmd_topic', '/legged/cmd_vel')
        cmd_topic = self.get_parameter('cmd_topic').get_parameter_value().string_value
        self.get_logger().info(f"[navRunner]: Command topic name: {cmd_topic}.")
        self.action_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.goal_vis_pub = self.create_publisher(MarkerArray, 'navigation_runner/goal', 10)

        # Service client
        raycast_client_group = MutuallyExclusiveCallbackGroup()
        self.raycast_client = self.create_client(RayCast, '/occupancy_map/raycast', callback_group=raycast_client_group)
        dynamic_obstacle_client_group = MutuallyExclusiveCallbackGroup()
        self.get_dyn_obs_client = self.create_client(GetDynamicObstacles, '/onboard_detector/get_dynamic_obstacles', callback_group=dynamic_obstacle_client_group)
        safe_action_client_group = MutuallyExclusiveCallbackGroup()
        self.get_safe_action_client = self.create_client(GetSafeAction, '/safe_action/get_safe_action', callback_group=safe_action_client_group)
        while not self.raycast_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('[navRunner]: Service /occupancy_map/raycast not available, waiting...')
        

        # Controller
        self.angle_controller = AnglePIDController(kp=1.0, ki=0.0, kd=0.1, dt=0.05, max_angular_velocity=1.0)

        # Load model
        self.declare_parameter('checkpoint_file', 'navrl_checkpoint.pt')
        ckpt_file = self.get_parameter('checkpoint_file').get_parameter_value().string_value
        self.get_logger().info(f"[navRunner]: Checkpoint: {ckpt_file}.")
        self.policy = self.init_model(ckpt_file)
        self.policy.eval()


    def init_model(self, ckpt_file):
        observation_dim = 8
        num_dim_each_dyn_obs_state = 10
        observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.cfg.device), 
                    "lidar": UnboundedContinuousTensorSpec((1, self.lidar_hbeams, self.cfg.sensor.lidar_vbeams), device=self.cfg.device),
                    "direction": UnboundedContinuousTensorSpec((1, 3), device=self.cfg.device),
                    "dynamic_obstacle": UnboundedContinuousTensorSpec((1, self.cfg.algo.feature_extractor.dyn_obs_num, num_dim_each_dyn_obs_state), device=self.cfg.device),
                }),
            }).expand(1)
        }, shape=[1], device=self.cfg.device)

        action_dim = 3
        action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": UnboundedContinuousTensorSpec((action_dim,), device=self.cfg.device), 
            })
        }).expand(1, action_dim).to(self.cfg.device)

        policy = PPO(self.cfg.algo, observation_spec, action_spec, self.cfg.device)
        file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpts")
        checkpoint = os.path.join(file_dir, ckpt_file)
        policy.load_state_dict(torch.load(checkpoint, weights_only=True))
        self.get_logger().info("[navRunner]: Model load successfully.")

        return policy

    def safety_check_callback(self, msg):
        if (msg.data == True):
            self.safety_stop = True
        else:
            self.safety_stop = False

    def odom_callback(self, odom):
        self.odom = odom
        self.odom_received = True

    def goal_callback(self, goal):
        if not self.odom_received:
            return

        self.goal = goal
        self.goal.pose.position.z = self.odom.pose.pose.position.z
        dir_x = self.goal.pose.position.x - self.odom.pose.pose.position.x
        dir_y = self.goal.pose.position.y - self.odom.pose.pose.position.y
        dir_z = self.goal.pose.position.z - self.odom.pose.pose.position.z
        self.target_dir = torch.tensor([dir_x, dir_y, dir_z], device=self.cfg.device) 

        self.goal_received = True

    def get_raycast(self, pos, start_angle):
        raypoints = []
        pos_msg = Point()
        pos_msg.x = pos[0]
        pos_msg.y = pos[1]
        pos_msg.z = pos[2]

        # Service request
        request = RayCast.Request()
        request.position = pos_msg
        request.start_angle = float(start_angle)
        request.range = float(self.cfg.sensor.lidar_range)
        request.vfov_min = float(self.cfg.sensor.lidar_vfov[0])
        request.vfov_max = float(self.cfg.sensor.lidar_vfov[1])
        request.vbeams = int(self.cfg.sensor.lidar_vbeams)
        request.hres = float(self.cfg.sensor.lidar_hres)
        request.visualize = self.vis_raycast

        # Call the service asynchronously
        response = self.raycast_client.call(request)
        num_points = int(len(response.points) / 3)
        self.laser_points_msg = response.points
        for i in range(num_points):
            p = [
                response.points[3 * i + 0],
                response.points[3 * i + 1],
                response.points[3 * i + 2]
            ]
            raypoints.append(p)

        return raypoints

    def get_dynamic_obstacles(self, pos):
        dynamic_obstacle_pos = torch.zeros(self.cfg.algo.feature_extractor.dyn_obs_num, 3, dtype=torch.float, device=self.cfg.device)
        dynamic_obstacle_vel = torch.zeros(self.cfg.algo.feature_extractor.dyn_obs_num, 3, dtype=torch.float, device=self.cfg.device)
        dynamic_obstacle_size = torch.zeros(self.cfg.algo.feature_extractor.dyn_obs_num, 3, dtype=torch.float, device=self.cfg.device)

        distance_range = 4.0
        pos_msg = Point()
        pos_msg.x = pos[0]
        pos_msg.y = pos[1]
        pos_msg.z = pos[2]
        
        # Create a request object
        request = GetDynamicObstacles.Request()
        request.current_position = pos_msg
        request.range = float(distance_range)

        # Call the service asynchronously
        response = self.get_dyn_obs_client.call(request)
        total_obs_num = len(response.position)
        max_obs_num = self.cfg.algo.feature_extractor.dyn_obs_num

        for i in range(min(max_obs_num, total_obs_num)):
            pos_vec = response.position[i]
            vel_vec = response.velocity[i]
            size_vec = response.size[i]

            dynamic_obstacle_pos[i] = torch.tensor(
                [pos_vec.x, pos_vec.y, pos_vec.z],
                dtype=torch.float, device=self.cfg.device
            )
            dynamic_obstacle_vel[i] = torch.tensor(
                [vel_vec.x, vel_vec.y, vel_vec.z],
                dtype=torch.float, device=self.cfg.device
            )
            dynamic_obstacle_size[i] = torch.tensor(
                [size_vec.x, size_vec.y, size_vec.z],
                dtype=torch.float, device=self.cfg.device
            )

        return dynamic_obstacle_pos, dynamic_obstacle_vel, dynamic_obstacle_size

    def check_obstacle(self, lidar_scan, dyn_obs_states):
        quarter_size = lidar_scan.shape[2] // 4
        first_quarter_check, last_quarter_check = torch.all(lidar_scan[:, :, :quarter_size, 1:] < 0.2), torch.all(lidar_scan[:, :, -quarter_size:, 1:] < 0.2)
        has_static = (not first_quarter_check) or (not last_quarter_check)
        has_dynamic = not torch.all(dyn_obs_states == 0.)
        return has_static or has_dynamic

    def get_safe_action(self, vel_world, action_vel_world):
        if self.get_safe_action_client.service_is_ready():
            safe_action = np.zeros(3)
            pos_msg = Point(x=self.odom.pose.pose.position.x, y=self.odom.pose.pose.position.y, z=self.odom.pose.pose.position.z)
            vel_msg = Vector3(x=vel_world[0].item(), y=vel_world[1].item(), z=vel_world[2].item())
            action_vel_msg = Vector3(x=action_vel_world[0].item(), y=action_vel_world[1].item(), z=action_vel_world[2].item())
            max_vel = np.sqrt(2. * self.vel_limit**2)
            obstacle_pos_list = []
            obstacle_vel_list = []
            obstacle_size_list = []
            for i in range(len(self.dynamic_obstacles[0])):
                if (self.dynamic_obstacles[2][i][0] != 0):
                    obs_pos = Vector3(x=self.dynamic_obstacles[0][i][0].item(), y=self.dynamic_obstacles[0][i][1].item(), z=self.dynamic_obstacles[0][i][2].item())
                    obs_vel = Vector3(x=self.dynamic_obstacles[1][i][0].item(), y=self.dynamic_obstacles[1][i][1].item(), z=self.dynamic_obstacles[1][i][2].item())
                    obs_size = Vector3(x=self.dynamic_obstacles[2][i][0].item(), y=self.dynamic_obstacles[2][i][1].item(), z=self.dynamic_obstacles[2][i][2].item())
                    obstacle_pos_list.append(obs_pos)
                    obstacle_vel_list.append(obs_vel)
                    obstacle_size_list.append(obs_size)

            # Service request
            request = GetSafeAction.Request()
            request.agent_position = pos_msg
            request.agent_velocity = vel_msg
            request.agent_size = self.robot_size
            request.obs_position = obstacle_pos_list
            request.obs_velocity = obstacle_vel_list
            request.obs_size = obstacle_size_list
            request.laser_points = self.laser_points_msg
            request.laser_range = float(self.cfg.sensor.lidar_range)
            request.laser_res = float(max(self.raycast_vres, self.raycast_hres))
            request.max_velocity = float(max_vel)
            request.rl_velocity = action_vel_msg

            # Call the service asynchronously
            response = self.get_safe_action_client.call(request)
            safe_action = np.array([
                response.safe_action.x,
                response.safe_action.y,
                response.safe_action.z
            ])
            return safe_action
        else:
            return action_vel_world

    def raycast_callback(self):
        # self.get_logger().info("[navRunner]: raycast callback start.")
        if not self.odom_received or not self.goal_received:
            return
        pos = np.array([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z])
        start_angle = np.arctan2(self.target_dir[1].cpu().numpy(), self.target_dir[0].cpu().numpy())
        self.raypoints = self.get_raycast(pos, start_angle)
        # self.get_logger().info("[navRunner]: raycast callback end.")


    def dynamic_obstacle_callback(self):
        # self.get_logger().info("[navRunner]: DO callback start.")
        if not self.odom_received:
            return
        pos = np.array([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z])
        dynamic_obstacle_pos, dynamic_obstacle_vel, dynamic_obstacle_size = self.get_dynamic_obstacles(pos)
        self.dynamic_obstacles = (dynamic_obstacle_pos, dynamic_obstacle_vel, dynamic_obstacle_size)
        # self.get_logger().info("[navRunner]: DO callback end.")


    def get_action(self, pos: torch.Tensor, vel: torch.Tensor, goal: torch.Tensor): # use world velocity
        rpos = goal - pos        
        distance = rpos.norm(dim=-1, keepdim=True)
        distance_2d = rpos[..., :2].norm(dim=-1, keepdim=True)
        distance_z = rpos[..., 2].unsqueeze(-1)


        target_dir_2d = self.target_dir.clone()
        target_dir_2d[2] = 0.

        rpos_clipped = rpos / distance.clamp(1e-6) # start to goal direction
        rpos_clipped_g = vec_to_new_frame(rpos_clipped, target_dir_2d).squeeze(0).squeeze(0)

        # "relative" velocity
        vel_g = vec_to_new_frame(vel, target_dir_2d).squeeze(0).squeeze(0) # goal velocity

        # drone_state = torch.cat([rpos_clipped, orientation, vel_g], dim=-1).squeeze(1)
        drone_state = torch.cat([rpos_clipped_g, distance_2d, distance_z, vel_g], dim=-1).unsqueeze(0)

        # Lidar States
        lidar_scan = torch.tensor(self.raypoints, device=self.cfg.device)
        lidar_scan = (lidar_scan - pos).norm(dim=-1).clamp_max(self.cfg.sensor.lidar_range).reshape(1, 1, self.lidar_hbeams, self.cfg.sensor.lidar_vbeams)
        lidar_scan = self.cfg.sensor.lidar_range - lidar_scan

        # dynamic obstacle states
        dynamic_obstacle_pos = self.dynamic_obstacles[0].clone()
        dynamic_obstacle_vel = self.dynamic_obstacles[1].clone()
        dynamic_obstacle_size = self.dynamic_obstacles[2].clone()
        closest_dyn_obs_rpos = dynamic_obstacle_pos - pos
        closest_dyn_obs_rpos[dynamic_obstacle_size[:, 2] == 0] = 0.
        closest_dyn_obs_rpos[:, 2][dynamic_obstacle_size[:, 2] > 1] = 0.
        closest_dyn_obs_rpos_g = vec_to_new_frame(closest_dyn_obs_rpos.unsqueeze(0), target_dir_2d).squeeze(0)
        closest_dyn_obs_distance = closest_dyn_obs_rpos.norm(dim=-1, keepdim=True)
        closest_dyn_obs_distance_2d = closest_dyn_obs_rpos_g[..., :2].norm(dim=-1, keepdim=True)
        closest_dyn_obs_distance_z = closest_dyn_obs_rpos_g[..., 2].unsqueeze(-1)
        closest_dyn_obs_rpos_gn = closest_dyn_obs_rpos_g / closest_dyn_obs_distance.clamp(1e-6)


        closest_dyn_obs_vel_g = vec_to_new_frame(dynamic_obstacle_vel.unsqueeze(0), target_dir_2d).squeeze(0)
        
        obs_res = 0.25
        closest_dyn_obs_width = torch.max(dynamic_obstacle_size[:, 0], dynamic_obstacle_size[:, 1])
        closest_dyn_obs_width += self.robot_size * 2.
        closest_dyn_obs_width = torch.clamp(torch.ceil(closest_dyn_obs_width / 0.25) - 1, min=0, max=1./obs_res - 1)
        closest_dyn_obs_width[dynamic_obstacle_size[:, 2] == 0] = 0.
        closest_dyn_obs_height = dynamic_obstacle_size[:, 2]
        closest_dyn_obs_height[(closest_dyn_obs_height <= 1) & (closest_dyn_obs_height != 0)] = 1.
        closest_dyn_obs_height[closest_dyn_obs_height > 1] = 0.
        # dyn_obs_states = torch.cat([closest_dyn_obs_rpos_g, closest_dyn_obs_vel_g, \
        #                             closest_dyn_obs_width.unsqueeze(1), closest_dyn_obs_height.unsqueeze(1)], dim=-1).unsqueeze(0).unsqueeze(0)
        dyn_obs_states = torch.cat([closest_dyn_obs_rpos_gn, closest_dyn_obs_distance_2d, closest_dyn_obs_distance_z, closest_dyn_obs_vel_g, \
                                    closest_dyn_obs_width.unsqueeze(1), closest_dyn_obs_height.unsqueeze(1)], dim=-1).unsqueeze(0).unsqueeze(0)
        # states
        obs = TensorDict({
            "agents": TensorDict({
                "observation": TensorDict({
                    "state": drone_state,
                    "lidar": lidar_scan,
                    "direction": target_dir_2d,
                    "dynamic_obstacle": dyn_obs_states
                })
            })
        })

        has_obstacle_in_range = self.check_obstacle(lidar_scan, dyn_obs_states)
        if (has_obstacle_in_range):
            with set_exploration_type(ExplorationType.MEAN):
                output = self.policy(obs)
            # vel_world = output["agents", "action"]
            vel_local_normalized = output["agents", "action_normalized"]
            vel_local_world = 2.0 * vel_local_normalized * self.vel_limit - self.vel_limit
            vel_world = vec_to_world(vel_local_world, output["agents", "observation", "direction"])
        else:
            vel_world = (goal - pos)/torch.norm(goal - pos) * self.vel_limit
        return vel_world

    def control_callback(self):
        # self.get_logger().info("[navRunner]: Control callback start.")

        if (not self.odom_received):
            return

        if (not self.goal_received or len(self.raypoints) == 0 or len(self.dynamic_obstacles) == 0):
            return

        if (self.safety_stop):
            final_cmd_vel = Twist()
            final_cmd_vel.linear.x = 0.
            final_cmd_vel.linear.y = 0.
            final_cmd_vel.angular.x = 0.
            self.action_pub.publish(final_cmd_vel)
            self.get_logger().info("[navRunner]: Emergency stop triggers.")
            return

        goal_angle = np.arctan2(self.target_dir[1].cpu().numpy(), self.target_dir[0].cpu().numpy())
        _, _, curr_angle = self.quaternion_to_euler(self.odom.pose.pose.orientation.w, self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y, self.odom.pose.pose.orientation.z)
        angle_diff = np.abs(goal_angle - curr_angle)
        angular_velocity = self.angle_controller.compute_angular_velocity(goal_angle, curr_angle)

        if (angle_diff >= 0.3):
            final_cmd_vel = Twist()
            final_cmd_vel.angular.z = angular_velocity
            self.action_pub.publish(final_cmd_vel)
            return

        pos = torch.tensor([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z], device=self.cfg.device)
        goal = torch.tensor([self.goal.pose.position.x, self.goal.pose.position.y, self.goal.pose.position.z], device=self.cfg.device)
        rot = self.quaternion_to_rotation_matrix(self.odom.pose.pose.orientation)
        vel_body = np.array([self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z])
        vel_world = torch.tensor(rot @ vel_body, device=self.cfg.device, dtype=torch.float) # world vel

        # get RL action from model
        cmd_vel_world = self.get_action(pos, vel_world, goal).squeeze(0).squeeze(0).detach().cpu().numpy()        
        self.cmd_vel_world = cmd_vel_world.copy()

        # get safe action
        safe_cmd_vel_world = self.get_safe_action(vel_world, cmd_vel_world)
        # safe_cmd_vel_world[2] = 0
        self.safe_cmd_vel_world = safe_cmd_vel_world.copy()
        quat_no_tilt = self.euler_to_quaternion(0, 0, curr_angle)
        quat_msg = Quaternion()
        quat_msg.w = quat_no_tilt[0]
        quat_msg.x = quat_no_tilt[1]
        quat_msg.y = quat_no_tilt[2]
        quat_msg.z = quat_no_tilt[3]
        rot_no_tilt = self.quaternion_to_rotation_matrix(quat_msg)
        safe_cmd_vel_local = np.linalg.inv(rot_no_tilt) @ safe_cmd_vel_world

        # Goal condition: 
        distance = (pos - goal).norm() 
        if (distance <= 3.0 and distance > 1.0):
            if (np.linalg.norm(safe_cmd_vel_local) != 0):
                safe_cmd_vel_local = 1.0 * safe_cmd_vel_local/np.linalg.norm(safe_cmd_vel_local)
                safe_cmd_vel_world = 1.0 * safe_cmd_vel_world/np.linalg.norm(safe_cmd_vel_world)
        elif (distance <= 1.0):
            safe_cmd_vel_local *= 0.
            safe_cmd_vel_world *= 0.
        
        # Send command
        final_cmd_vel = Twist()
        final_cmd_vel.linear.x = safe_cmd_vel_local[0].item()
        final_cmd_vel.linear.y = safe_cmd_vel_local[1].item()
        final_cmd_vel.angular.z = angular_velocity
        if (self.height_control):
            final_cmd_vel.linear.z = safe_cmd_vel_world[2].item()
        else:
            final_cmd_vel.linear.z = 0.0
        self.action_pub.publish(final_cmd_vel)
        # self.get_logger().info("[navRunner]: Control callback end.")


    def euler_to_quaternion(self, roll, pitch, yaw):
        # Compute the half angles
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        # Calculate the quaternion components
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return (w, x, y, z)
    
    def quaternion_to_euler(self, w, x, y, z):
        # Normalize the quaternion
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        w /= norm
        x /= norm
        y /= norm
        z /= norm

        # Roll (X-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (Y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            # Use 90 degrees if out of range
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (Z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return (roll, pitch, yaw)
    
    def quaternion_to_rotation_matrix(self, quaternion):
        # w, x, y, z = quaternion
        w = quaternion.w
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        xx, xy, xz = x**2, x*y, x*z
        yy, yz = y**2, y*z
        zz = z**2
        wx, wy, wz = w*x, w*y, w*z
        
        return np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
        ])  

    def run(self):
        raycast_cb_group = MutuallyExclusiveCallbackGroup()
        dynamic_obstacle_cb_group = MutuallyExclusiveCallbackGroup()
        control_cb_group = MutuallyExclusiveCallbackGroup()
        vis_cb_group = MutuallyExclusiveCallbackGroup()
        self.create_timer(0.05, self.raycast_callback, callback_group=raycast_cb_group)
        self.create_timer(0.05, self.dynamic_obstacle_callback, callback_group=dynamic_obstacle_cb_group)
        self.create_timer(0.05, self.control_callback, callback_group=control_cb_group)
        self.create_timer(0.05, self.goal_vis_callback, callback_group=vis_cb_group)
        self.get_logger().info("[navRunner]: Start running!")

    def goal_vis_callback(self):
        if not self.goal_received:
            return
        msg = MarkerArray()
        goal_point = Marker()
        goal_point.header.frame_id = "map"
        goal_point.header.stamp = self.get_clock().now().to_msg()
        goal_point.ns = "goal_point"
        goal_point.id = 1
        goal_point.type = goal_point.SPHERE
        goal_point.action = goal_point.ADD
        goal_point.pose.position.x = self.goal.pose.position.x
        goal_point.pose.position.y = self.goal.pose.position.y
        goal_point.pose.position.z = self.goal.pose.position.z
        goal_point.lifetime = Duration(sec=0, nanosec=int(0.1 * 1e9))
        goal_point.scale.x = 0.3
        goal_point.scale.y = 0.3
        goal_point.scale.z = 0.3
        goal_point.color.r = 1.0
        goal_point.color.b = 1.0
        goal_point.color.a = 1.0
        msg.markers.append(goal_point)
        self.goal_vis_pub.publish(msg)