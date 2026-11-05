cmake_minimum_required(VERSION 3.0.2)
project(moxibustion_api)

find_package(catkin REQUIRED COMPONENTS
  acupoint_detection
  actionlib
  actionlib_msgs
  geometry_msgs
  message_generation
  moxa_robot_interface
  rospy
  std_msgs
  std_srvs
)

catkin_python_setup()

add_message_files(FILES SensorSample.msg)
add_action_files(FILES MoxibustionTask.action)
generate_messages(DEPENDENCIES
  acupoint_detection
  actionlib_msgs
  geometry_msgs
  std_msgs
)

catkin_package(CATKIN_DEPENDS
  acupoint_detection
  actionlib
  actionlib_msgs
  geometry_msgs
  message_runtime
  moxa_robot_interface
  rospy
  std_msgs
  std_srvs
)

catkin_install_python(PROGRAMS
  scripts/moxibustion_action_server.py
  scripts/serial_sensor_node.py
  scripts/send_task.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  PATTERN "__pycache__" EXCLUDE
)
