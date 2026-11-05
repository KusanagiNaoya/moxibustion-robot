"""整机界面的单任务 Action 客户端。"""
import threading

import actionlib
import rospy

from moxibustion_api.msg import MoxibustionTaskAction, MoxibustionTaskGoal


class MoxibustionActionClient:
    def __init__(self, action_name, wait_timeout):
        self.client = actionlib.SimpleActionClient(action_name, MoxibustionTaskAction)
        self.wait_timeout = wait_timeout
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._submitted = False

    def prepare(self):
        with self._lock:
            self._cancel.clear()
            self._submitted = False

    def send_goal(self, acupoints, target_name, mode, duration, target_temperature,
                  done_cb, active_cb, feedback_cb):
        """在工作线程等待服务；返回 False 表示发送前已取消。"""
        if not self.client.wait_for_server(rospy.Duration(self.wait_timeout)):
            raise RuntimeError('艾灸 Action 服务不可用')
        goal = MoxibustionTaskGoal()
        goal.acupoints = acupoints
        goal.target_name = target_name
        goal.mode = mode
        goal.duration = duration
        goal.target_temperature = target_temperature
        # 仅串行化短暂提交；取消不等待服务连接或机器人移动。
        with self._lock:
            if self._cancel.is_set():
                return False
            self._submitted = True
            self.client.send_goal(goal, done_cb=done_cb, active_cb=active_cb,
                                  feedback_cb=feedback_cb)
        return True

    def cancel_goal(self):
        self._cancel.set()
        with self._lock:
            if self._submitted:
                self.client.cancel_goal()
