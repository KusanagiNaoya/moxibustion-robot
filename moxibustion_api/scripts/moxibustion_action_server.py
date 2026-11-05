#!/usr/bin/env python3
"""启动艾灸 Action 服务器，等待任务请求。"""
import rospy
from moxibustion_api.task_server import MoxibustionActionServer


def main():
    rospy.init_node('moxibustion_action_server')
    MoxibustionActionServer()
    rospy.spin()


if __name__ == '__main__':
    main()
