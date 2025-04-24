/*
	FILE: safe_action_node.cpp
	--------------------------------------
	safe action ROS2 Node
*/
#include <rclcpp/rclcpp.hpp>
#include <navigation_runner/safeAction.h>

int main(int argc, char* argv[]){
    rclcpp::init(argc, argv);
    std::shared_ptr<navigationRunner::safeAction> sa = std::make_shared<navigationRunner::safeAction>();
    rclcpp::spin(sa);
    rclcpp::shutdown();
    return 0;
}