#!/usr/bin/env python3
import rospy
from acupoint_detection.model_servers import HRNetServer


if __name__ == '__main__':
    rospy.init_node('hrnet_keypoints_server')
    server = HRNetServer()
    rospy.spin()
