#!/usr/bin/env python3

import rclpy
from yolo_detector import yolo_detector

def main(args=None):
    rclpy.init(args=args)
    yolo_node = yolo_detector()
    rclpy.spin(yolo_node)
    yolo_node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
	main()
	
