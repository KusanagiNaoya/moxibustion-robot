cmake_minimum_required(VERSION 3.0.2)
project(acupoint_detection)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  message_filters
  message_generation
  rospy
  sensor_msgs
  std_msgs
  std_srvs
)

catkin_python_setup()

add_message_files(FILES
  Acupoint.msg
  AcupointArray.msg
  Keypoint.msg
)
add_service_files(FILES
  AcupointDetection.srv
  HRNetKeypoints.srv
  UNetSegmentation.srv
)
generate_messages(DEPENDENCIES geometry_msgs sensor_msgs std_msgs)

catkin_package(CATKIN_DEPENDS
  cv_bridge geometry_msgs message_filters message_runtime rospy sensor_msgs std_msgs std_srvs
)

catkin_install_python(PROGRAMS
  scripts/acupoint_detection_server_node.py
  scripts/hrnet_server_node.py
  scripts/unet_server_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY config launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
