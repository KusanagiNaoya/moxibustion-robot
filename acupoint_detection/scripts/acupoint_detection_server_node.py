#!/usr/bin/env python3
import rospy
from acupoint_detection.detection_server import AcupointDetectionServer


if __name__ == '__main__':
    rospy.init_node('acupoint_detection_server')
    server = AcupointDetectionServer()
    rospy.spin()
