/*
	FILE: esdf_map_node.cpp
	--------------------------------------
	ESDF map ROS2 Node
*/
#include <rclcpp/rclcpp.hpp>
#include <map_manager/ESDFMap.h>

int main(int argc, char* argv[]){
    rclcpp::init(argc, argv);
    std::shared_ptr<mapManager::ESDFMap> map = std::make_shared<mapManager::ESDFMap>();
    map->initMap();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(map);
    executor.spin();    
    rclcpp::shutdown();
    return 0;
}