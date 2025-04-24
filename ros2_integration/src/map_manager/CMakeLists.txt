cmake_minimum_required(VERSION 3.8)
project(map_manager)

# Set C++ standard and compile options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compile options
add_compile_options(-O3 -Wall)


# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED QUIET)
find_package(pcl_conversions REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(onboard_detector REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/CheckPosCollision.srv"
  "srv/GetStaticObstacles.srv"
  "srv/RayCast.srv"
  DEPENDENCIES geometry_msgs
)

# Package library
add_library(${PROJECT_NAME}_lib SHARED
  include/${PROJECT_NAME}/raycast/raycast.cpp
  include/${PROJECT_NAME}/clustering/Kmeans.cpp
  include/${PROJECT_NAME}/clustering/obstacleClustering.cpp
  include/${PROJECT_NAME}/occupancyMap.cpp
  include/${PROJECT_NAME}/ESDFMap.cpp
)
target_include_directories(${PROJECT_NAME}_lib
    PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)
set(dependencies "ament_cmake" "rclcpp" "message_filters" "sensor_msgs" "geometry_msgs"
    "cv_bridge" "Eigen3" "PCL" "pcl_conversions" "nav_msgs" "visualization_msgs" "onboard_detector")
ament_target_dependencies(${PROJECT_NAME}_lib ${dependencies})
rosidl_get_typesupport_target(cpp_typesupport_target "${PROJECT_NAME}" "rosidl_typesupport_cpp")
target_link_libraries(${PROJECT_NAME}_lib "${cpp_typesupport_target}")

# Nodes
add_executable(occupancy_map_node src/occupancy_map_node.cpp)
target_link_libraries(occupancy_map_node ${PROJECT_NAME}_lib) 

add_executable(esdf_map_node src/esdf_map_node.cpp)
target_link_libraries(esdf_map_node ${PROJECT_NAME}_lib) 

# headers for client packages
install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

# library for client packages
install(TARGETS ${PROJECT_NAME}_lib
  EXPORT ${PROJECT_NAME}_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# install executables
install(TARGETS
  occupancy_map_node
  esdf_map_node
  DESTINATION lib/${PROJECT_NAME})

# install config
install(
  DIRECTORY cfg/
  DESTINATION share/${PROJECT_NAME}/cfg
)

# install launch
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# install rviz config
install(
  DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
)

ament_export_targets(${PROJECT_NAME}_lib HAS_LIBRARY_TARGET)
ament_export_dependencies(${dependencies})
ament_package()
