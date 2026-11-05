"""统一机器人接口客户端；位姿使用米和四元数。"""
import math
import threading

import rospy
from geometry_msgs.msg import PoseStamped
from moxa_robot_interface.msg import RobotState
from moxa_robot_interface.srv import MoveToPose
from std_srvs.srv import Trigger

from moxibustion_api.control import PoseSample


class RobotClient:
    def __init__(self, base_frame, interface_ns):
        self.base_frame = base_frame
        self.move_service = rospy.names.ns_join(interface_ns, 'move_to_pose')
        self.stop_service = rospy.names.ns_join(interface_ns, 'stop')
        state_topic = rospy.names.ns_join(interface_ns, 'state')
        self._lock = threading.Lock()
        self._sample = None
        self._error = '尚无机器人反馈'
        self._move = rospy.ServiceProxy(self.move_service, MoveToPose)
        self._subscriber = rospy.Subscriber(state_topic, RobotState, self._receive, queue_size=1)

    def _receive(self, message):
        p, q = message.pose.pose.position, message.pose.pose.orientation
        pose = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        # 包装消息可能重发缓存；保留底层反馈接收时间。
        stamp = message.pose.header.stamp.to_sec()
        error = ''
        if not message.valid:
            error = '机器人反馈无效'
        elif not message.controlled_frame.strip():
            error = '机器人反馈缺少受控参考点'
        elif message.pose.header.frame_id != self.base_frame:
            error = '机器人反馈坐标系与 base_frame 不一致'
        elif stamp <= 0 or not all(math.isfinite(v) for v in (*pose, stamp)):
            error = '机器人反馈位姿或时间无效'
        elif abs(math.hypot(*pose[3:]) - 1.0) > 1e-3:
            error = '机器人反馈四元数不是单位四元数'
        with self._lock:
            self._error = error
            self._sample = None if error else PoseSample(stamp, pose)

    def snapshot(self):
        """返回 PoseSample 或 None，由调用者检查采样时间。"""
        with self._lock:
            return self._sample

    def require_fresh_pose(self, now_s, max_age_s):
        with self._lock:
            sample, error = self._sample, self._error
        if sample is None:
            raise ValueError(error or '尚无机器人反馈')
        if not 0 <= now_s - sample.stamp_s <= max_age_s:
            raise ValueError('机器人原始反馈接收时间已过期或倒退')
        return sample

    def connect(self):
        rospy.wait_for_service(self.move_service, timeout=1.0)
        rospy.wait_for_service(self.stop_service, timeout=1.0)

    def move(self, pose):
        """同步提交 (x, y, z, qx, qy, qz, qw)，返回后端成功标志。"""
        result = self._move(self.pose_message(pose))
        if not result.success:
            rospy.logwarn('点目标接口返回失败：%s', result.message)
        return result.success

    def stop(self):
        """返回停止请求的处理结果，不表示机器人已物理静止。"""
        try:
            rospy.wait_for_service(self.stop_service, timeout=1.0)
            # 独立代理允许取消线程与运动调用并行。
            result = rospy.ServiceProxy(self.stop_service, Trigger)()
            return result.success, result.message
        except (rospy.ROSException, rospy.ServiceException) as exc:
            return False, '停止接口异常：' + str(exc)

    def pose_message(self, pose, stamp_s=0.0):
        """反馈保留采样时刻；点目标时间为零，未知位姿以 NaN 表示。"""
        values = pose if pose is not None else (float('nan'),) * 7
        message = PoseStamped()
        message.header.frame_id = self.base_frame
        message.header.stamp = rospy.Time.from_sec(stamp_s if pose is not None else 0.0)
        message.pose.position.x, message.pose.position.y, message.pose.position.z = values[:3]
        (message.pose.orientation.x, message.pose.orientation.y,
         message.pose.orientation.z, message.pose.orientation.w) = values[3:]
        return message
