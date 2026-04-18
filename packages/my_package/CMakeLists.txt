cmake_minimum_required(VERSION 2.8.3)
project(my_package)

find_package(catkin REQUIRED COMPONENTS
  duckietown_msgs
  rospy
  sensor_msgs
  std_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  src/camera_reader_node.py
  src/heading_control_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
