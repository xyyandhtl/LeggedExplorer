/*
	FILE: occupancy_map_node.cpp
	--------------------------------------
	occupancy map ROS2 Node
*/
#include <rclcpp/rclcpp.hpp>
#include <map_manager/occupancyMap.h>

int main(int argc, char* argv[]){
    rclcpp::init(argc, argv);
    std::shared_ptr<mapManager::occMap> map = std::make_shared<mapManager::occMap>();
    map->initMap();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(map);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}