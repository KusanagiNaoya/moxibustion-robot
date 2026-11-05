#!/usr/bin/env python3
import rospy
from acupoint_detection.model_servers import UNetServer


if __name__ == '__main__':
    rospy.init_node('unet_segmentation_server')
    server = UNetServer()
    rospy.spin()
