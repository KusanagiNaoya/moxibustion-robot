#!/usr/bin/env python3
"""Kinova Gen3 Lite 的 ROS1 点目标、Home、夹爪和停止服务。"""

import threading
import time

import rospy
from kortex_driver.msg import (
    ActionEvent, ActionNotification, BaseCyclic_Feedback, CartesianReferenceFrame,
    CartesianSpeed, Finger, GripperMode,
)
from kortex_driver.srv import (
    Base_ClearFaults, ExecuteAction, ExecuteActionRequest,
    OnNotificationActionTopic, OnNotificationActionTopicRequest,
    PlayCartesianTrajectory, PlayCartesianTrajectoryRequest,
    ReadAction, ReadActionRequest, SendGripperCommand, SendGripperCommandRequest,
    SetCartesianReferenceFrame, SetCartesianReferenceFrameRequest, Stop, StopRequest,
)
from std_srvs.srv import Trigger, TriggerResponse

from kinova_controller.msg import ControllerStatus
from kinova_controller.srv import (
    GripperService, GripperServiceResponse,
    MoveCartesianService, MoveCartesianServiceResponse,
)


class RobotControllerNode:
    """移动、Home 和夹爪共用命令锁，停止请求独立处理。"""

    def __init__(self):
        rospy.init_node('robot_controller_node', anonymous=True)
        self.robot_name = rospy.get_param('~robot_name', 'my_gen3_lite').strip('/')
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        self.motion_timeout = rospy.get_param('~motion_timeout', 5.0)
        self.home_timeout = rospy.get_param('~home_timeout', 10.0)
        self.action_poll_dt = rospy.get_param('~action_poll_dt', 0.01)
        self.default_translation_speed = rospy.get_param('~default_translation_speed', 1.0)
        self.default_orientation_speed = rospy.get_param('~default_orientation_speed', 15.0)
        self.command_lock = threading.Lock()
        self.action_notif_lock = threading.Lock()
        self.feedback_lock = threading.Lock()
        self.last_action_notif_type = None
        self.robot_feedback = None
        self.feedback_received_at = None
        self.HOME_ACTION_IDENTIFIER = 2

        self._init_kinova_services()
        self.stop_motion = rospy.ServiceProxy(f'/{self.robot_name}/base/stop', Stop)
        self.stop_robot_service = rospy.Service(
            '/stop_robot_service', Trigger, self._handle_stop_robot_request)
        self.robot_status_sub = rospy.Subscriber(
            f'/{self.robot_name}/base_feedback', BaseCyclic_Feedback,
            self._robot_status_callback)
        self.controller_status_pub = rospy.Publisher(
            '/controller_status', ControllerStatus, queue_size=10)
        self.controller_status_timer = rospy.Timer(
            rospy.Duration(0.1), self._publish_controller_status)

        if self.kinova_available:
            faults_cleared = self._clear_faults()
            notifications_enabled = self._subscribe_to_robot_notification()
            reference_configured = self._set_cartesian_reference_frame()
            self.kinova_available = all((
                faults_cleared, notifications_enabled, reference_configured))
        if not self.kinova_available:
            rospy.logerr('Kinova 初始化未完成：仅保留状态和停止，未注册移动、Home 或夹爪服务。')
            return

        # 初始化成功后才开放运动服务，避免请求与参考方式设置并发。
        self.move_cartesian_service = rospy.Service(
            '/move_cartesian_service', MoveCartesianService, self._handle_move_cartesian_request)
        self.gripper_service = rospy.Service(
            '/gripper_service', GripperService, self._handle_gripper_request)
        self.home_robot_service = rospy.Service(
            '/home_robot_service', Trigger, self._handle_home_robot_request)

    def _init_kinova_services(self):
        service_configs = [
            ('/base/clear_faults', Base_ClearFaults, 'clear_faults'),
            ('/base/execute_action', ExecuteAction, 'execute_action'),
            ('/base/play_cartesian_trajectory', PlayCartesianTrajectory, 'play_cartesian'),
            ('/base/read_action', ReadAction, 'read_action'),
            ('/base/send_gripper_command', SendGripperCommand, 'send_gripper_command'),
            ('/base/activate_publishing_of_action_topic', OnNotificationActionTopic,
             'activate_publishing_of_action_notification'),
            ('/control_config/set_cartesian_reference_frame', SetCartesianReferenceFrame,
             'set_cartesian_reference_frame'),
        ]
        try:
            for suffix, service_type, attribute in service_configs:
                name = f'/{self.robot_name}{suffix}'
                rospy.wait_for_service(name, timeout=5.0)
                setattr(self, attribute, rospy.ServiceProxy(name, service_type))
            self.action_topic_sub = rospy.Subscriber(
                f'/{self.robot_name}/action_topic', ActionNotification,
                self._action_notification_callback)
            self.kinova_available = True
        except rospy.ROSException as exc:
            rospy.logerr('连接 Kinova 服务失败：%s', exc)
            self.kinova_available = False

    def _action_notification_callback(self, notification):
        with self.action_notif_lock:
            self.last_action_notif_type = notification.action_event

    def _set_last_action_notif_type(self, value):
        with self.action_notif_lock:
            self.last_action_notif_type = value

    def _wait_for_action_end_or_abort(self, timeout=None):
        """仅 ACTION_END 返回 True；等待超时不会自动发送停止。"""
        if timeout is None:
            timeout = self.motion_timeout
        start_time = time.time()
        while not rospy.is_shutdown():
            with self.action_notif_lock:
                notification = self.last_action_notif_type
            if notification == ActionEvent.ACTION_END:
                return True
            if notification == ActionEvent.ACTION_ABORT:
                return False
            if time.time() - start_time > timeout:
                rospy.logerr('等待动作通知超时：%.2f s；尚未确认机械臂停止', timeout)
                return False
            time.sleep(self.action_poll_dt)
        return False

    def _subscribe_to_robot_notification(self):
        try:
            self.activate_publishing_of_action_notification(OnNotificationActionTopicRequest())
        except rospy.ServiceException as exc:
            rospy.logerr('启用动作通知失败：%s', exc)
            return False
        rospy.sleep(1.0)
        return True

    def _clear_faults(self):
        if not self.kinova_available:
            return False
        try:
            self.clear_faults()
        except rospy.ServiceException as exc:
            rospy.logerr('清除故障失败：%s', exc)
            return False
        return True

    def _set_cartesian_reference_frame(self):
        if not self.kinova_available:
            return False
        request = SetCartesianReferenceFrameRequest()
        request.input.reference_frame = CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED
        try:
            self.set_cartesian_reference_frame(request)
        except rospy.ServiceException as exc:
            rospy.logerr('设置参考方式失败：%s', exc)
            return False
        rospy.sleep(0.25)
        return True

    def _home_robot(self):
        if not self.kinova_available:
            return False
        with self.command_lock:
            if not self._clear_faults():
                rospy.logwarn('清故障失败，继续尝试 Home')
            self._set_last_action_notif_type(None)
            read_request = ReadActionRequest()
            read_request.input.identifier = self.HOME_ACTION_IDENTIFIER
            try:
                action = self.read_action(read_request)
                execute_request = ExecuteActionRequest()
                execute_request.input = action.output
                self.execute_action(execute_request)
            except rospy.ServiceException as exc:
                rospy.logerr('Home 调用失败：%s', exc)
                return False
            return self._wait_for_action_end_or_abort(self.home_timeout)

    def _move_to_pose(self, x, y, z, theta_x, theta_y, theta_z):
        """发送绝对点目标；xyz 为 m，theta_* 为 Kortex 度制欧拉角。"""
        if not self.kinova_available:
            return False
        with self.command_lock:
            if not self._clear_faults():
                rospy.logwarn('清故障失败，继续尝试移动')
            request = PlayCartesianTrajectoryRequest()
            request.input.target_pose.x = x
            request.input.target_pose.y = y
            request.input.target_pose.z = z
            request.input.target_pose.theta_x = theta_x
            request.input.target_pose.theta_y = theta_y
            request.input.target_pose.theta_z = theta_z
            speed = CartesianSpeed()
            speed.translation = self.default_translation_speed
            speed.orientation = self.default_orientation_speed
            request.input.constraint.oneof_type.speed.append(speed)
            self._set_last_action_notif_type(None)
            try:
                self.play_cartesian(request)
            except rospy.ServiceException as exc:
                rospy.logerr('笛卡尔移动调用失败：%s', exc)
                return False
            return self._wait_for_action_end_or_abort(self.motion_timeout)

    def _send_gripper_command(self, value):
        """发送 0–1 夹爪位置；成功不表示已确认到位。"""
        if not self.kinova_available:
            return False
        with self.command_lock:
            if not self._clear_faults():
                rospy.logwarn('清故障失败，继续尝试夹爪命令')
            request = SendGripperCommandRequest()
            finger = Finger()
            finger.finger_identifier = 0
            finger.value = value
            request.input.gripper.finger.append(finger)
            request.input.mode = GripperMode.GRIPPER_POSITION
            try:
                self.send_gripper_command(request)
            except rospy.ServiceException as exc:
                rospy.logerr('夹爪调用失败：%s', exc)
                return False
            time.sleep(0.5)
            return True

    def _handle_move_cartesian_request(self, request):
        success = self._move_to_pose(
            request.x, request.y, request.z,
            request.theta_x, request.theta_y, request.theta_z)
        return MoveCartesianServiceResponse(success=success)

    def _handle_gripper_request(self, request):
        return GripperServiceResponse(success=self._send_gripper_command(request.value))

    def _handle_home_robot_request(self, request):
        success = self._home_robot()
        message = 'Home Action 已完成' if success else 'Home 失败、被中止或超时'
        return TriggerResponse(success=success, message=message)

    def _handle_stop_robot_request(self, request):
        """绕过命令锁请求停止；success 只表示停止服务返回。"""
        try:
            rospy.wait_for_service(f'/{self.robot_name}/base/stop', timeout=1.0)
            self.stop_motion(StopRequest())
            return TriggerResponse(True, 'Kortex 停止服务已返回，实际停止效果以反馈为准')
        except (rospy.ROSException, rospy.ServiceException) as exc:
            return TriggerResponse(False, 'Kortex 停止调用失败：' + str(exc))

    def _robot_status_callback(self, message):
        received_at = rospy.Time.now()
        with self.feedback_lock:
            self.robot_feedback = message
            self.feedback_received_at = received_at

    def _publish_controller_status(self, event=None):
        """重发缓存时保留反馈接收时间，frame_id 仅标注坐标系。"""
        with self.feedback_lock:
            feedback = self.robot_feedback
            received_at = self.feedback_received_at
        if feedback is None:
            return
        message = ControllerStatus()
        message.header.stamp = received_at
        message.header.frame_id = self.base_frame
        message.robot_status = feedback
        message.pose.x = feedback.base.tool_pose_x
        message.pose.y = feedback.base.tool_pose_y
        message.pose.z = feedback.base.tool_pose_z
        message.pose.roll = feedback.base.tool_pose_theta_x
        message.pose.pitch = feedback.base.tool_pose_theta_y
        message.pose.yaw = feedback.base.tool_pose_theta_z
        self.controller_status_pub.publish(message)


def main():
    try:
        RobotControllerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
