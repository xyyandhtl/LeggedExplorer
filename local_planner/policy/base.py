import time
import threading
import numpy as np


class LocalPlannerBase:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.policy_device
        self.num_envs = cfg.num_envs
        self.update_dt = 1. / cfg.local_planner_freq
        print(f'local_planner update dt: {self.update_dt}')
        self.rgb_data = [],
        self.depth_data = [],
        self.scan_data = None,
        self.ego_state = {}
        self.goal_info = { 'goal_type': '',    # todo: 'blind', 'waypoint', 'path'
                           'goal_cmd': [],     # for blind goal, like just walking forward
                           'goal_position': [],
                           'goal_orientation': [], }
        self.rgb_cameras = None
        self.depth_cameras = None
        self.commands = np.zeros((self.num_envs, 3))   # default stand still    # x vel, y vel, yaw vel, (heading)

        # self.goal_info['goal_type'] = self.cfg.goal_type
        # if self.cfg.goal_type == 'blind':
        #     self.goal_info['goal_cmd'] = self.cfg.goal_cmd
        # elif self.cfg.goal_type == 'waypoint':
        #     self.goal_info['goal_position'] = self.cfg.goal_position
        #     self.goal_info['goal_orientation'] = self.cfg.goal_orientation
        # else:
        #     raise NotImplementedError("Goal type not implemented")

        self._running = False
        self._thread = None

    def get_current_cmd_vel(self):
        return self.commands

    def update_sensor_data(self):
        raise NotImplementedError("Subclasses must implement this method")

    def update_ego_state(self):
        raise NotImplementedError("Subclasses must implement this method")

    def update_goal_info(self):
        raise NotImplementedError("Subclasses must implement this method")

    def infer_action(self):
        raise NotImplementedError("Subclasses must implement this method")

    def start(self):
        raise NotImplementedError("Subclasses must implement this method")

    def stop(self):
        raise NotImplementedError("Subclasses must implement this method")

    def reset(self):
        self.rgb_data = [],
        self.depth_data = [],
        self.scan_data = None,
        self.ego_state = {}


class LocalPlannerIsaac(LocalPlannerBase):
    def set_rgb_cameras(self, rgb_cameras):
        self.rgb_cameras = rgb_cameras

    def set_depth_cameras(self, depth_cameras):
        self.depth_cameras = depth_cameras

    def update_sensor_data(self):
        if self.rgb_cameras:
            self.rgb_data = []
            for i, camera in enumerate(self.rgb_cameras):
                rgb = camera.get_rgb()
                self.rgb_data.append(rgb)
        if self.depth_cameras:
            self.depth_data = []
            for i, camera in enumerate(self.depth_cameras):
                depth = camera.get_depth()
                self.depth_data.append(depth)

    def update_goal_info(self):
        # todo
        return

    def update_ego_state(self):
        # todo
        return

    def infer_action(self):
        raise NotImplementedError("Subclasses must implement this method")

    def run(self):
        while True:
            # todo: use loop thread or direct inference call
            self.update_sensor_data()
            self.update_ego_state()
            self.update_goal_info()
            self.infer_action()
            time.sleep(self.update_dt)

    def start(self):
        if self._thread is None:
            self._running = True
            self._thread = threading.Thread(target=self.run, daemon=True)
            self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None



from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class LocalPlannerROS2(LocalPlannerBase, Node):
    def __init__(self, cfg):
        LocalPlannerBase.__init__(self, cfg)
        Node.__init__(self, "local_planner_ros2_node")
        self.rgb_images = {}
        self.depth_images = {}
        self.bridge = CvBridge()

    def set_rgb_cameras(self, rgb_topics):
        self.rgb_cameras = rgb_topics
        for topic in rgb_topics:
            self.create_subscription(Image, topic, self._rgb_callback_factory(topic), 10)

    def set_depth_cameras(self, depth_topics):
        self.depth_cameras = depth_topics
        for topic in depth_topics:
            self.create_subscription(Image, topic, self._depth_callback_factory(topic), 10)

    def _rgb_callback_factory(self, topic_name):
        def callback(msg):
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.rgb_images[topic_name] = cv_image
            except Exception as e:
                self.get_logger().error(f"Failed to convert RGB image: {e}")
        return callback

    def _depth_callback_factory(self, topic_name):
        def callback(msg):
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                self.depth_images[topic_name] = cv_image
            except Exception as e:
                self.get_logger().error(f"Failed to convert Depth image: {e}")
        return callback

    def update_sensor_data(self):
        """
        Copy latest received images into self.sensor_data.
        """
        self.rgb_data = [self.rgb_images.get(topic) for topic in self.rgb_cameras]
        self.depth_data = [self.depth_images.get(topic) for topic in self.depth_cameras]

    def infer_action(self):
        raise NotImplementedError("Subclasses must implement this method")