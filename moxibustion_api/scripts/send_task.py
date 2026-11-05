#!/usr/bin/env python3
"""向艾灸 Action 提交真实运动任务。"""
import argparse

import actionlib
import rospy
from acupoint_detection.msg import Acupoint, AcupointArray
from moxibustion_api.msg import MoxibustionTaskAction, MoxibustionTaskGoal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', default='/moxibustion_task')
    parser.add_argument('--frame', default='base_link')
    parser.add_argument('--name', default='manual_target')
    parser.add_argument('--mode', choices=('pid_control', 'sac_pid_control'), default='')
    parser.add_argument('--x', type=float, required=True, help='穴位基座 X，m')
    parser.add_argument('--y', type=float, required=True, help='穴位基座 Y，m')
    parser.add_argument('--z', type=float, required=True, help='穴位表面 Z，m；准备 Z 由服务器配置')
    parser.add_argument('--temperature', type=float, required=True, help='目标滤波平均温度，°C')
    parser.add_argument('--duration', type=float, required=True, help='控制秒数')
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node('send_moxibustion_task', anonymous=True, disable_signals=True)
    client = actionlib.SimpleActionClient(args.action, MoxibustionTaskAction)
    if not client.wait_for_server(rospy.Duration(5.0)):
        raise RuntimeError('任务服务器不可用')
    points = AcupointArray()
    points.header.frame_id = args.frame
    points.header.stamp = rospy.Time.now()
    points.is_3d = True
    point = Acupoint()
    point.name, point.is_3d, point.confidence = args.name, True, 1.0
    point.point.x, point.point.y, point.point.z = args.x, args.y, args.z
    points.acupoints = [point]
    goal = MoxibustionTaskGoal()
    goal.acupoints, goal.target_name, goal.mode = points, args.name, args.mode
    goal.duration, goal.target_temperature = args.duration, args.temperature

    def show_feedback(message):
        rospy.loginfo_throttle(1.0, '%s T=%.3f°C r=%.4fm h=%.4fm %s',
                              message.mode, message.current_temperature,
                              message.current_distance, message.current_h, message.message)

    try:
        client.send_goal(goal, feedback_cb=show_feedback)
        while not client.wait_for_result(rospy.Duration(0.2)):
            pass
        rospy.loginfo('Action状态=%s，结果=%s', client.get_state(), client.get_result())
    except KeyboardInterrupt:
        client.cancel_goal()
        if client.wait_for_result(rospy.Duration(5.0)):
            rospy.loginfo('取消后结果：%s', client.get_result())
        else:
            rospy.logwarn('已发送取消但未收到终态')


if __name__ == '__main__':
    main()
