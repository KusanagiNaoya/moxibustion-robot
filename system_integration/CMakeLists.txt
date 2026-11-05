cmake_minimum_required(VERSION 3.0.2)
project(system_integration)

find_package(catkin REQUIRED COMPONENTS
  acupoint_detection actionlib actionlib_msgs cv_bridge geometry_msgs message_filters
  moxa_robot_interface moxibustion_api rospy sensor_msgs std_srvs tf tf2_geometry_msgs tf2_ros
)
catkin_python_setup()
catkin_package(CATKIN_DEPENDS
  acupoint_detection actionlib actionlib_msgs cv_bridge geometry_msgs message_filters
  moxa_robot_interface moxibustion_api rospy sensor_msgs std_srvs tf tf2_geometry_msgs tf2_ros
)
catkin_install_python(PROGRAMS
  scripts/panel_acupoint_moxibustion.py
  scripts/publish_hand_eye_tf.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
