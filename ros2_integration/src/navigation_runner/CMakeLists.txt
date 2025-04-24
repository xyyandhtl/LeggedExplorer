cmake_minimum_required(VERSION 3.8)
project(navigation_runner)

# Set C++ standard and compile options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compile options
add_compile_options(-O3 -Wall)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(onboard_detector REQUIRED)
find_package(map_manager REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetSafeAction.srv"
  DEPENDENCIES geometry_msgs
)

# Package library
add_library(${PROJECT_NAME}_lib SHARED
  include/${PROJECT_NAME}/safeAction.cpp
)
target_include_directories(${PROJECT_NAME}_lib
    PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)
set(dependencies "ament_cmake" "rclcpp" "onboard_detector" "visualization_msgs" "geometry_msgs")
ament_target_dependencies(${PROJECT_NAME}_lib ${dependencies})
rosidl_get_typesupport_target(cpp_typesupport_target "${PROJECT_NAME}" "rosidl_typesupport_cpp")
target_link_libraries(${PROJECT_NAME}_lib "${cpp_typesupport_target}")

# Nodes
add_executable(safe_action_node src/safe_action_node.cpp)
target_link_libraries(safe_action_node ${PROJECT_NAME}_lib)

# install executables
install(TARGETS
  safe_action_node
  DESTINATION lib/${PROJECT_NAME}
)

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
  scripts/navigation_node.py
  scripts/cfg
  DESTINATION lib/${PROJECT_NAME}
)

# install rviz config
install(
  DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
)

ament_package()
