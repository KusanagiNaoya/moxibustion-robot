#!/usr/bin/env python3
"""将 Kinova 点目标和反馈接入统一机器人接口。"""

import copy
import math
import threading

import rospy
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger, TriggerResponse
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from kinova_controller.msg import ControllerStatus
from kinova_controller.srv import GripperService, MoveCartesianService
from moxa_robot_interface.msg import RobotState
from moxa_robot_interface.srv import (
    MoveToPose, MoveToPoseResponse, SetGripperPosition, SetGripperPositionResponse,
)


class KinovaInterfaceNode:
    def __init__(self):
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        self.controlled_frame = rospy.get_param('~controlled_frame', 'kortex_tool')
        self.feedback_timeout_s = float(rospy.get_param('~feedback_timeout_s', 1.0))
        interface_ns = rospy.get_param('~robot_interface_ns', '/moxa_robot')
        if not self.base_frame or not self.controlled_frame or not interface_ns:
            raise ValueError('坐标系、受控点标识和接口命名空间不能为空')
        if not math.isfinite(self.feedback_timeout_s) or self.feedback_timeout_s <= 0:
            raise ValueError('feedback_timeout_s 必须为有限正数')
        self.move_service = rospy.get_param('~move_service', '/move_cartesian_service')
        self.stop_service = rospy.get_param('~stop_service', '/stop_robot_service')
        status_topic = rospy.get_param('~status_topic', '/controller_status')
        prefix = interface_ns.rstrip('/')
        self._state_lock = threading.Lock()
        self._last_stamp_s = 0.0
        self._state = self._invalid_state()
        self.state_pub = rospy.Publisher(prefix + '/state', RobotState, queue_size=10)
        self.status_sub = rospy.Subscriber(
            status_topic, ControllerStatus, self._status_callback, queue_size=1)
        self.stop_server = rospy.Service(prefix + '/stop', Trigger, self._stop)
        self.state_timer = rospy.Timer(rospy.Duration(0.1), self._publish_state)
        # 底层完成初始化后才开放移动；等待期间状态和停止仍可用。
        rospy.wait_for_service(self.move_service)
        self.move_server = rospy.Service(
            prefix + '/move_to_pose', MoveToPose, self._move_to_pose)
        if rospy.get_param('~enable_home', False):
            self.home_service = rospy.get_param('~home_service', '/home_robot_service')
            self.home_server = rospy.Service(prefix + '/home', Trigger, self._home)
        if rospy.get_param('~enable_gripper', False):
            self.gripper_service = rospy.get_param('~gripper_service', '/gripper_service')
            self.gripper_server = rospy.Service(
                prefix + '/set_gripper_position', SetGripperPosition, self._set_gripper)

    def _invalid_state(self):
        state = RobotState()
        state.pose = PoseStamped()
        state.pose.header.frame_id = self.base_frame
        state.controlled_frame = self.controlled_frame
        state.valid = False
        state.pose.pose.position.x = float('nan')
        state.pose.pose.position.y = float('nan')
        state.pose.pose.position.z = float('nan')
        state.pose.pose.orientation.x = float('nan')
        state.pose.pose.orientation.y = float('nan')
        state.pose.pose.orientation.z = float('nan')
        state.pose.pose.orientation.w = float('nan')
        return state

    def _status_callback(self, message):
        state = self._invalid_state()
        state.pose.header = copy.deepcopy(message.header)
        stamp_s = message.header.stamp.to_sec()
        pose = message.pose
        values = (pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw)
        valid = (message.header.frame_id == self.base_frame
                 and math.isfinite(stamp_s) and stamp_s > 0
                 and all(math.isfinite(value) for value in values))
        if valid:
            # Kortex 度制 xyz 外旋：Rz(yaw) Ry(pitch) Rx(roll)。
            quaternion = quaternion_from_euler(
                math.radians(pose.roll), math.radians(pose.pitch),
                math.radians(pose.yaw), axes='sxyz')
            state.pose.pose.position.x = pose.x
            state.pose.pose.position.y = pose.y
            state.pose.pose.position.z = pose.z
            orientation = state.pose.pose.orientation
            orientation.x, orientation.y, orientation.z, orientation.w = quaternion
            valid = all(math.isfinite(value) for value in quaternion)
        with self._state_lock:
            if not math.isfinite(stamp_s) or stamp_s < self._last_stamp_s:
                valid = False
            elif stamp_s > 0:
                self._last_stamp_s = stamp_s
            state.valid = valid
            self._state = state

    def _publish_state(self, event):
        with self._state_lock:
            state = copy.deepcopy(self._state)
        age = (rospy.Time.now() - state.pose.header.stamp).to_sec()
        state.valid = (state.valid and math.isfinite(age)
                       and 0 <= age <= self.feedback_timeout_s)
        # 缓存重发保留反馈接收时间；当前时间仅用于判断时效。
        self.state_pub.publish(state)

    def _move_to_pose(self, request):
        target = request.target
        if target.header.frame_id != self.base_frame:
            return MoveToPoseResponse(False, '目标坐标系必须为 ' + self.base_frame)
        position = target.pose.position
        orientation = target.pose.orientation
        xyz = (position.x, position.y, position.z)
        quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
        if not all(math.isfinite(value) for value in xyz + quaternion):
            return MoveToPoseResponse(False, '目标位姿包含非有限数值')
        norm = math.hypot(*quaternion)
        if norm == 0 or abs(norm - 1.0) > 1e-3:
            return MoveToPoseResponse(False, '目标姿态必须为单位四元数')
        quaternion = tuple(value / norm for value in quaternion)
        angles = tuple(math.degrees(value) for value in
                       euler_from_quaternion(quaternion, axes='sxyz'))
        try:
            rospy.wait_for_service(self.move_service, timeout=1.0)
            move = rospy.ServiceProxy(self.move_service, MoveCartesianService)
            result = move(*(xyz + angles))
        except (rospy.ROSException, rospy.ServiceException) as exc:
            return MoveToPoseResponse(False, '底层移动服务调用失败：' + str(exc))
        if result.success:
            return MoveToPoseResponse(True, '底层移动返回成功')
        return MoveToPoseResponse(False, '底层移动返回失败')

    def _stop(self, request):
        # 独立代理且不获取运动锁，停止请求无需等待同步移动返回。
        try:
            rospy.wait_for_service(self.stop_service, timeout=1.0)
            stop = rospy.ServiceProxy(self.stop_service, Trigger)
            result = stop()
            return TriggerResponse(result.success, result.message)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            return TriggerResponse(False, '底层停止服务调用失败：' + str(exc))

    def _home(self, request):
        try:
            rospy.wait_for_service(self.home_service, timeout=1.0)
            result = rospy.ServiceProxy(self.home_service, Trigger)()
            return TriggerResponse(result.success, result.message)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            return TriggerResponse(False, '底层 Home 服务调用失败：' + str(exc))

    def _set_gripper(self, request):
        if not math.isfinite(request.position) or not 0.0 <= request.position <= 1.0:
            return SetGripperPositionResponse(False, '夹爪目标必须为 [0, 1] 内的有限数')
        try:
            rospy.wait_for_service(self.gripper_service, timeout=1.0)
            result = rospy.ServiceProxy(self.gripper_service, GripperService)(request.position)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            return SetGripperPositionResponse(False, '底层夹爪服务调用失败：' + str(exc))
        return SetGripperPositionResponse(
            result.success, '夹爪目标请求已处理，未确认到位' if result.success else '底层夹爪返回失败')


def main():
    rospy.init_node('robot_interface_node')
    try:
        KinovaInterfaceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
