"""通过 Action 编排 PID/SAC-PID 艾灸任务。"""
from dataclasses import dataclass
import math
import threading
import time

import actionlib
import rospy

from moxibustion_api.control import (
    ControlFault, DistanceLoop, DistanceSample, SampleRejected,
    TemperatureController, TemperatureSample, TimingLimits,
)
from moxibustion_api.msg import (
    MoxibustionTaskAction, MoxibustionTaskFeedback, MoxibustionTaskResult, SensorSample,
)
from moxibustion_api.robot_adapter import RobotClient


METHODS = {'pid_control': 'pid', 'sac_pid_control': 'sac_pid'}


@dataclass
class TaskRun:
    method: str
    duration_s: float
    target_c: float
    phase: str = 'PREPARING'
    started_s: object = None
    exit_kind: str = ''
    reason: str = ''
    motion_submitted: bool = False
    stop_thread: object = None
    early_stop_result: object = None
    command_pose: object = None
    motion_result: str = 'NOT_SUBMITTED'


class TaskExit(Exception):
    """中断任务步骤，由统一出口设置终态。"""


class DistanceJumpGuard:
    """检测无法由机器人 Z 位移解释的距离突变。"""

    def __init__(self, parameters):
        self.enabled = parameters['enable_dist_jump_guard']
        self.threshold = parameters['dist_jump_threshold']
        self.residual_threshold = parameters['dist_jump_residual_threshold']
        self.confirm = parameters['dist_jump_confirm_count']
        self.warmup = parameters['dist_jump_warmup_cycles']
        if (not isinstance(self.enabled, bool)
                or not all(math.isfinite(v) and v > 0 for v in
                           (self.threshold, self.residual_threshold))
                or not all(isinstance(v, int) and not isinstance(v, bool) and v >= 1
                           for v in (self.confirm, self.warmup))):
            raise ValueError('距离突变检测的开关、阈值或周期数无效')
        self.previous = None
        self.count = self.cycles = 0

    def observe(self, raw_distance_m, robot_z_m):
        """每个有效新距离调用一次；确认突变时抛出 ControlFault。"""
        if not self.enabled:
            return
        if self.previous is None:
            self.previous = (raw_distance_m, robot_z_m)
            self.cycles = 1
            return
        distance_delta = raw_distance_m - self.previous[0]
        # 用独立的机器人 Z 反馈计算残差；传感器派生的间距会使残差恒为零。
        residual = distance_delta - (robot_z_m - self.previous[1])
        if self.cycles < self.warmup:
            self.cycles += 1
            self.count = 0
        elif abs(distance_delta) >= self.threshold and abs(residual) >= self.residual_threshold:
            self.count += 1
            if self.count >= self.confirm:
                raise ControlFault('距离突变：Δr={:.4f} m，Δr-Δz={:.4f} m'.format(
                    distance_delta, residual))
        else:
            self.count = 0
        self.previous = (raw_distance_m, robot_z_m)


class MoxibustionActionServer:
    """方法在任务开始前确定，PID/SAC-PID 共用任务流程。"""

    def __init__(self):
        self.p = rospy.get_param('~')
        self.data_lock = threading.Lock()
        self.task_lock = threading.RLock()
        self.sensor = self.thermal = self.current = None
        self.temperature_floor_s = 0.0
        self.robot = RobotClient(self.p['base_frame'], self.p['robot_interface_ns'])
        self.sensor_sub = rospy.Subscriber(
            self.p['sensor_topic'], SensorSample, self.sensor_callback, queue_size=1)
        self.server = actionlib.SimpleActionServer(
            self.p['action_name'], MoxibustionTaskAction,
            execute_cb=self.execute, auto_start=False)
        self.server.register_preempt_callback(self.preempt)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.monitor)
        rospy.on_shutdown(self.shutdown)
        self.server.start()
        rospy.loginfo('艾灸 Action 待机：%s；默认方法=%s',
                      self.p['action_name'], self.p['default_mode'])

    def sensor_callback(self, message):
        """保留温度更新时的配对距离，按通道时间戳识别新采样。"""
        with self.data_lock:
            self.sensor = message
            if (message.temperature_valid
                    and message.temperature_stamp.to_sec() > self.temperature_floor_s
                    and (self.thermal is None
                         or message.temperature_stamp > self.thermal.temperature_stamp)):
                self.thermal = message

    def inputs(self):
        with self.data_lock:
            return self.sensor, self.thermal

    def distance_sample(self, message):
        """将传感器米值换算为实际间距，安装偏置仅在此补偿。"""
        return DistanceSample(message.distance_stamp.to_sec(),
                              message.distance_m - self.p['D_setup'] + self.p['D_raw'])

    def select_goal(self, goal):
        method = goal.mode if goal.mode else self.p['default_mode']
        if method not in METHODS:
            raise ValueError('未知方法 {!r}，仅支持 pid_control/sac_pid_control'.format(method))
        if not all(math.isfinite(v) for v in (goal.duration, goal.target_temperature)) or goal.duration < 0:
            raise ValueError('目标温度和时长须有限，时长须 >= 0')
        points = goal.acupoints
        if points.header.frame_id != self.p['base_frame'] or not points.is_3d:
            raise ValueError('任务必须提供 base_frame 中的三维穴位；本入口不做 TF 转换')
        matches = [p for p in points.acupoints if p.name == goal.target_name]
        if len(matches) != 1:
            raise ValueError('target_name 必须唯一匹配一个穴位')
        point = matches[0]
        if (not point.is_3d or not all(math.isfinite(v) for v in
                (point.point.x, point.point.y, point.point.z, point.confidence))
                or point.confidence < 0):
            raise ValueError('穴位坐标或置信度无效')
        for value, lower, upper, label in (
                (goal.target_temperature, 'target_temperature_min_c', 'target_temperature_max_c', '目标温度'),
                (point.point.x, 'x_min_m', 'x_max_m', '目标 X'),
                (point.point.y, 'y_min_m', 'y_max_m', '目标 Y')):
            low, high = float(self.p[lower]), float(self.p[upper])
            if not (math.isfinite(low) and math.isfinite(high) and low <= high):
                raise ValueError(label + '范围配置无效')
            if not low <= value <= high:
                raise ValueError('{}超出配置范围 [{}, {}]'.format(label, low, high))
        return TaskRun(method, goal.duration, goal.target_temperature), point.point

    def configure(self, task):
        """每任务新建控制器；SAC 初始化失败须在准备运动前返回。"""
        names = ('initial_z_m', 'reference_rate_hz', 'control_rate_hz',
                 'pid_kp', 'pid_ki', 'pid_kd', 'D_setup', 'D_raw',
                 'distance_min_m', 'distance_max_m', 'z_min_m', 'z_max_m',
                 'distance_gain', 'deadband_m', 'max_delta_z_m', 'min_dt_s',
                 'max_dt_s', 'max_age_s', 'max_skew_s', 'inference_timeout_s',
                 'max_reference_age_s', 'input_timeout_s', 'local_enter_c', 'local_exit_c')
        for name in names:
            if not math.isfinite(float(self.p[name])):
                raise ValueError(name + ' 必须是有限数')
        if self.p['control_rate_hz'] <= 0 or self.p['input_timeout_s'] <= 0:
            raise ValueError('循环频率和输入等待时间须 > 0')
        if not self.p['z_min_m'] <= self.p['initial_z_m'] <= self.p['z_max_m']:
            raise ValueError('准备 Z 超出配置范围')
        if not 0 < self.p['local_enter_c'] <= self.p['local_exit_c']:
            raise ValueError('温区进入/退出阈值须为正且有序')
        return TemperatureController(
            METHODS[task.method], task.target_c,
            (self.p['distance_min_m'], self.p['distance_max_m']),
            TimingLimits(self.p['min_dt_s'], self.p['max_dt_s'],
                         self.p['max_age_s'], self.p['max_skew_s']),
            model_dir=self.p.get('model_dir'),
            pid_gains=(self.p['pid_kp'], self.p['pid_ki'], self.p['pid_kd']),
            reference_rate_hz=self.p['reference_rate_hz'],
            inference_timeout_s=self.p['inference_timeout_s'],
            local_enter_c=self.p['local_enter_c'], local_exit_c=self.p['local_exit_c'])

    def wait_inputs(self, task):
        deadline = time.monotonic() + self.p['input_timeout_s']
        reason = '等待传感器'
        while time.monotonic() < deadline:
            self.check_exit(task)
            sensor, thermal = self.inputs()
            now = rospy.Time.now().to_sec()
            try:
                pose = self.robot.require_fresh_pose(now, self.p['max_age_s'])
                if (sensor is None or thermal is None or not sensor.temperature_valid
                        or not sensor.distance_valid or not thermal.distance_valid):
                    raise ValueError('尚无有效温度和距离')
                ages = (now - thermal.temperature_stamp.to_sec(),
                        now - thermal.distance_stamp.to_sec(), now - sensor.distance_stamp.to_sec())
                skew = thermal.temperature_stamp.to_sec() - thermal.distance_stamp.to_sec()
                if (not all(0 <= age <= self.p['max_age_s'] for age in ages)
                        or not 0 <= skew <= self.p['max_skew_s']):
                    raise ValueError('温度/距离样本过期或不能配对')
                if not all(math.isfinite(v) for v in
                           (thermal.temp_filtered_c, thermal.distance_m, sensor.distance_m)):
                    raise ValueError('温度或距离不是有限数')
                if not self.p['z_min_m'] <= pose.pose[2] <= self.p['z_max_m']:
                    raise ValueError('当前机器人 Z 超出配置范围')
                return pose
            except ValueError as exc:
                reason = str(exc)
            self.feedback(task, reason)
            # ROS 时钟尚未启动时，也需定期检查取消。
            time.sleep(0.05)
        raise ControlFault('准备输入超时：' + reason)

    def make_distance_loop(self, xy, orientation):
        """任务期间固定 XY 和姿态，仅沿基座 Z 调距。"""
        return DistanceLoop(
            xy, orientation, (self.p['z_min_m'], self.p['z_max_m']),
            gain=self.p['distance_gain'], max_delta_z_m=self.p['max_delta_z_m'],
            deadband_m=self.p['deadband_m'], max_age_s=self.p['max_age_s'],
            max_skew_s=self.p['max_skew_s'], max_reference_age_s=self.p['max_reference_age_s'],
            positive_z_increases_distance=self.p['positive_z_increases_distance'])

    def move(self, task, pose):
        """串行提交同步点目标，停止请求可并行发送。"""
        self.feedback(task, '准备提交点目标')
        self.check_exit(task)
        with self.task_lock:
            if task.exit_kind:
                raise TaskExit(task.reason)
            task.motion_submitted = True
            task.command_pose = pose
            task.motion_result = 'SUBMITTED'
        try:
            success = self.robot.move(pose)
        except rospy.ServiceException as exc:
            task.motion_result = 'CALL_ERROR'
            raise ControlFault('运动服务调用异常：' + str(exc))
        task.motion_result = 'SUCCEEDED' if success else 'FAILED'
        self.check_exit(task)
        if not success:
            raise ControlFault('点目标返回失败；不能据此认定已停止')

    def control_loop(self, task, outer, inner, guard):
        next_outer_s = 0.0
        last_accepted = time.monotonic()
        rate = rospy.Rate(self.p['control_rate_hz'])
        while not rospy.is_shutdown():
            self.check_exit(task)
            try:
                sensor, thermal = self.inputs()
                if (sensor is None or thermal is None or not sensor.temperature_valid
                        or not sensor.distance_valid):
                    raise SampleRejected('等待准备移动之后的新温度/距离')
                now = rospy.Time.now().to_sec()
                if now >= next_outer_s and thermal.distance_valid:
                    try:
                        result = outer.step(
                            TemperatureSample(thermal.temperature_stamp.to_sec(), thermal.temp_filtered_c),
                            self.distance_sample(thermal), now)
                    except SampleRejected:
                        pass
                    else:
                        inner.update_reference(result)
                        next_outer_s = now + 1.0 / self.p['reference_rate_hz']
                # 外环使用温度采样时的距离，内环使用最新距离和位姿。
                sensor, _ = self.inputs()
                pose = self.robot.snapshot()
                if pose is None or not sensor.temperature_valid or not sensor.distance_valid:
                    raise SampleRejected('距离或机器人位姿不可用')
                candidate = inner.step(self.distance_sample(sensor), pose, rospy.Time.now().to_sec())
                guard.observe(sensor.distance_m, pose.pose[2])
                self.check_exit(task)
                if candidate.commanded_delta_z_m != 0.0:
                    self.move(task, candidate.command_pose)
                else:
                    task.motion_result = 'NO_MOTION'
                # 输入等待超时不计入同步移动耗时。
                last_accepted = time.monotonic()
                self.feedback(task, '温区=' + outer.mode)
            except SampleRejected as exc:
                if time.monotonic() - last_accepted > self.p['input_timeout_s']:
                    raise ControlFault('有效控制输入等待超时：' + str(exc))
                self.feedback(task, '等待新鲜输入：' + str(exc))
            rate.sleep()
        self.request_exit(task, 'CANCELED', 'ROS 正在关闭')
        self.check_exit(task)

    def execute(self, goal):
        task = None
        try:
            task, point = self.select_goal(goal)
            with self.task_lock:
                self.current = task
            self.check_exit(task)
            rospy.loginfo('采用控制方法 %s，目标平均温度 %.3f °C', task.method, task.target_c)
            self.feedback(task, '初始化控制方法')
            outer = self.configure(task)
            guard = DistanceJumpGuard(self.p)
            self.check_exit(task)
            self.robot.connect()
            pose = self.wait_inputs(task)
            inner = self.make_distance_loop((point.x, point.y), pose.pose[3:])
            # 准备高度由配置指定，控制间距由传感器测量。
            initial = (point.x, point.y, self.p['initial_z_m']) + pose.pose[3:]
            self.move(task, initial)
            prepared_s = rospy.Time.now().to_sec()
            with self.data_lock:
                self.temperature_floor_s = prepared_s
                self.thermal = None  # 控制须等待准备完成后的新温度采样。
            with self.task_lock:
                task.started_s = prepared_s
                task.phase = 'RUNNING'
            self.control_loop(task, outer, inner, guard)
        except TaskExit:
            pass
        except Exception as exc:
            if task is None:
                self.server.set_aborted(MoxibustionTaskResult(False, str(exc)), str(exc))
                return
            with self.task_lock:
                if not task.exit_kind:
                    task.exit_kind, task.reason = 'FAILED', str(exc)
            rospy.logerr('艾灸任务失败：%s', exc)
        finally:
            if task is not None:
                self.finish(task)

    def check_exit(self, task):
        if rospy.is_shutdown():
            self.request_exit(task, 'CANCELED', 'ROS 正在关闭')
        elif self.server.is_preempt_requested():
            self.request_exit(task, 'CANCELED', '客户端取消或新任务请求抢占')
        elif (task.started_s is not None and task.duration_s > 0
              and rospy.Time.now().to_sec() - task.started_s >= task.duration_s):
            self.request_exit(task, 'COMPLETED', '任务达到设置时长')
        with self.task_lock:
            if task.exit_kind:
                raise TaskExit(task.reason)

    def request_exit(self, task, kind, reason):
        """保留首次退出原因，独立线程允许在运动调用阻塞时请求停止。"""
        with self.task_lock:
            if self.current is not task or task.exit_kind:
                return
            task.exit_kind, task.reason, task.phase = kind, reason, 'STOPPING'
            if task.motion_submitted:
                task.stop_thread = threading.Thread(target=self.early_stop, args=(task,), daemon=True)
                task.stop_thread.start()

    def early_stop(self, task):
        task.early_stop_result = self.robot.stop()

    def preempt(self):
        with self.task_lock:
            task = self.current
        if task is not None:
            self.request_exit(task, 'CANCELED', '客户端取消或新任务请求抢占')

    def monitor(self, event):
        """运动调用阻塞期间仍能按时请求停止。"""
        with self.task_lock:
            task = self.current
            expired = (task is not None and task.started_s is not None
                       and task.duration_s > 0
                       and rospy.Time.now().to_sec() - task.started_s >= task.duration_s)
        if expired:
            self.request_exit(task, 'COMPLETED', '任务达到设置时长')

    def finish(self, task):
        """等待在途调用并再次停止后设置终态；阻塞的 RPC 会延迟任务结束。"""
        task.phase = 'STOPPING'
        if task.stop_thread is not None:
            task.stop_thread.join()
        stop_ok, detail = True, '未提交机器人运动'
        if task.motion_submitted:
            stop_ok, detail = self.robot.stop()
        with self.task_lock:
            if not task.exit_kind:
                task.exit_kind, task.reason = 'FAILED', '任务意外退出'
            reason = '{}；停止结果：{}'.format(task.reason, detail)
            if not stop_ok:
                task.exit_kind = 'FAILED'
            task.phase = task.exit_kind
            kind = task.exit_kind
        # 抢占回调持有 Actionlib 锁；调用 Action API 前释放 task_lock 以免锁顺序反转。
        self.feedback(task, reason)
        result = MoxibustionTaskResult(kind == 'COMPLETED' and stop_ok, reason)
        if kind == 'COMPLETED' and stop_ok:
            self.server.set_succeeded(result, reason)
        elif kind == 'CANCELED' and stop_ok:
            self.server.set_preempted(result, reason)
        else:
            self.server.set_aborted(result, reason)
        with self.task_lock:
            self.current = None

    def feedback(self, task, detail):
        sensor, _ = self.inputs()
        pose = self.robot.snapshot()
        now = rospy.Time.now().to_sec()
        try:
            max_age = float(self.p['max_age_s'])
            if not math.isfinite(max_age) or max_age <= 0:
                max_age = 0.0
        except (KeyError, TypeError, ValueError):
            max_age = 0.0
        temperature_fresh = (sensor is not None and sensor.temperature_valid
                             and 0 <= now - sensor.temperature_stamp.to_sec() <= max_age)
        distance_fresh = (sensor is not None and sensor.distance_valid
                          and 0 <= now - sensor.distance_stamp.to_sec() <= max_age)
        pose_fresh = pose is not None and 0 <= now - pose.stamp_s <= max_age
        msg = MoxibustionTaskFeedback()
        msg.mode = task.method
        msg.target_temperature = task.target_c
        msg.elapsed = rospy.Duration(0 if task.started_s is None else
                                     max(0.0, rospy.Time.now().to_sec() - task.started_s))
        msg.message = '{} | {} | {}'.format(task.phase, task.motion_result, detail)
        msg.current_temperature = sensor.temp_filtered_c if temperature_fresh else float('nan')
        msg.current_distance = sensor.distance_m if distance_fresh else float('nan')
        try:
            msg.current_h = msg.current_distance - float(self.p['D_setup']) + float(self.p['D_raw'])
        except (KeyError, TypeError, ValueError):
            msg.current_h = float('nan')  # 初始化失败时配置可能不完整。
        msg.current_pose = self.robot.pose_message(
            pose.pose if pose_fresh else None, pose.stamp_s if pose_fresh else 0.0)
        msg.target_pose = self.robot.pose_message(task.command_pose)
        self.server.publish_feedback(msg)

    def shutdown(self):
        """在 rospy 关闭标志阻止服务等待之前，完成停止请求。"""
        with self.task_lock:
            task = self.current
        if task is not None:
            self.request_exit(task, 'CANCELED', 'ROS 正在关闭')
            if task.stop_thread is not None:
                task.stop_thread.join()
            if task.motion_submitted:
                self.robot.stop()
