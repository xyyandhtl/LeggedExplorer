#!/usr/bin/env python3

import rclpy
from rclpy.executors import MultiThreadedExecutor
from navigation_navrl import Navigation
from hydra import compose, initialize

RELATIVE_FILE_PATH = "cfg"
def main(args=None):
    rclpy.init(args=args)

    # Initialize Hydra with the config path
    with initialize(config_path=RELATIVE_FILE_PATH, version_base=None):
        cfg = compose(config_name="train")
    nav = Navigation(cfg)
    nav.run()
    executor = MultiThreadedExecutor()
    executor.add_node(nav)
    executor.spin()
    # rclpy.spin(nav)
    nav.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()