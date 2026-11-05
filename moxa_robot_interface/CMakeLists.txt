cmake_minimum_required(VERSION 3.0.2)
project(moxa_robot_interface)

find_package(catkin REQUIRED COMPONENTS geometry_msgs message_generation std_srvs)
add_message_files(FILES RobotState.msg)
add_service_files(FILES MoveToPose.srv SetGripperPosition.srv)
generate_messages(DEPENDENCIES geometry_msgs)
catkin_package(CATKIN_DEPENDS geometry_msgs message_runtime std_srvs)
