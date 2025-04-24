cmake_minimum_required(VERSION 3.8)
project(onboard_detector)

# Set C++ standard and compile options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compile options
add_compile_options(-O3 -Wall)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(PCL REQUIRED QUIET)
find_package(pcl_conversions REQUIRED)
find_package(rosidl_default_generators REQUIRED)


rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetDynamicObstacles.srv"
   DEPENDENCIES geometry_msgs
)

# Package library
add_library(${PROJECT_NAME}_lib SHARED
  include/${PROJECT_NAME}/detectors/dbscan.cpp
  include/${PROJECT_NAME}/detectors/uvDetector.cpp
  include/${PROJECT_NAME}/tracking/kalmanFilter.cpp
  include/${PROJECT_NAME}/dynamicDetector.cpp
)
target_include_directories(${PROJECT_NAME}_lib
    PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)
set(dependencies "ament_cmake" "rclcpp" "visualization_msgs" "sensor_msgs"
                 "cv_bridge" "tf2" "tf2_ros" "tf2_geometry_msgs" "geometry_msgs" "message_filters"
                 "nav_msgs" "vision_msgs" "PCL" "pcl_conversions")
ament_target_dependencies(${PROJECT_NAME}_lib ${dependencies})
rosidl_get_typesupport_target(cpp_typesupport_target "${PROJECT_NAME}" "rosidl_typesupport_cpp")
target_link_libraries(${PROJECT_NAME}_lib "${cpp_typesupport_target}")


# Nodes
add_executable(dynamic_detector_node src/dynamic_detector_node.cpp)
target_link_libraries(dynamic_detector_node ${PROJECT_NAME}_lib)


# headers for client packages
install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

# library for client packages
install(TARGETS ${PROJECT_NAME}_lib
  EXPORT ${PROJECT_NAME}_lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

# install executables
install(TARGETS
  dynamic_detector_node
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

# Install Python executables
install(PROGRAMS
  scripts/yolo_detector_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# install rviz config
install(
  DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
)

ament_export_targets(${PROJECT_NAME}_lib HAS_LIBRARY_TARGET)
ament_export_dependencies(${dependencies})
ament_package()
