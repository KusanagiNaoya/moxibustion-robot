cmake_minimum_required(VERSION 3.0.2)
project(kinova_controller)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs kortex_driver message_generation moxa_robot_interface rospy std_msgs std_srvs tf
)
add_message_files(FILES KinovaPose.msg ControllerStatus.msg)
add_service_files(FILES MoveCartesianService.srv GripperService.srv)
generate_messages(DEPENDENCIES kortex_driver std_msgs)
catkin_package(CATKIN_DEPENDS geometry_msgs kortex_driver message_runtime moxa_robot_interface rospy std_msgs std_srvs tf)

catkin_install_python(PROGRAMS scripts/robot_controller_node.py scripts/robot_interface_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
