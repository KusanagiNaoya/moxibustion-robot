"""穴位检测、任务操作和机器人手动操作台。"""
import copy
import math
import time

import cv2
import rospy
from cv_bridge import CvBridge
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QScrollArea, QSpinBox, QTabWidget,
    QTextEdit, QVBoxLayout, QWidget,
)
import pyqtgraph as pg
from actionlib_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import Point, PoseStamped
from std_srvs.srv import Trigger
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from acupoint_detection.msg import Acupoint, AcupointArray
from moxa_robot_interface.msg import RobotState
from moxa_robot_interface.srv import MoveToPose, SetGripperPosition
from moxibustion_api.msg import MoxibustionTaskActionResult, SensorSample
from system_integration.acupoint_workflow import AcupointWorkflowManager, ManualAcupointPicker
from system_integration.action_client import MoxibustionActionClient


class CallWorker(QThread):
    result = pyqtSignal(str, object, object)

    def __init__(self, key, call):
        super().__init__()
        self.key, self.call = key, call

    def run(self):
        try:
            self.result.emit(self.key, self.call(), None)
        except Exception as exc:
            self.result.emit(self.key, None, str(exc))


class ImageView(QLabel):
    pixel_clicked = pyqtSignal(int, int)

    def __init__(self):
        super().__init__('等待图像')
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 280)
        self.original = None
        self.pixel_size = None
        self.picking = False

    def show_bgr(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy()
        self.original = QPixmap.fromImage(image)
        self.pixel_size = (width, height)
        self._scale_image()

    def _scale_image(self):
        if self.original is not None:
            self.setPixmap(self.original.scaled(self.size(), Qt.KeepAspectRatio,
                                                Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_image()

    def mousePressEvent(self, event):
        if not self.picking or event.button() != Qt.LeftButton or self.pixmap() is None:
            return
        pixmap = self.pixmap()
        left = (self.width() - pixmap.width()) / 2
        top = (self.height() - pixmap.height()) / 2
        x, y = event.x() - left, event.y() - top
        if 0 <= x < pixmap.width() and 0 <= y < pixmap.height():
            width, height = self.pixel_size
            self.pixel_clicked.emit(int(x * width / pixmap.width()),
                                    int(y * height / pixmap.height()))


class AcupointMoxibustionPanel(QMainWindow):
    feedback_signal = pyqtSignal(object)
    done_signal = pyqtSignal(int, object)
    active_signal = pyqtSignal()
    robot_signal = pyqtSignal(object)
    sensor_signal = pyqtSignal(object)
    status_signal = pyqtSignal(object)
    result_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.task = rospy.get_param('~task')
        self.base_frame = self.task['base_frame']
        self.interface_ns = self.task['robot_interface_ns'].rstrip('/')
        self.service_timeout = float(rospy.get_param('~service_timeout', 3.0))
        self.status_timeout = float(rospy.get_param('~status_timeout', 3.0))
        self.max_age = float(self.task['max_age_s'])
        self.enable_home = bool(rospy.get_param('~enable_home', False))
        self.enable_gripper = bool(rospy.get_param('~enable_gripper', False))
        self.bridge = CvBridge()
        self.workflow = AcupointWorkflowManager(
            service_name=rospy.get_param('~detect_service'),
            ready_service=rospy.get_param('~ready_service'), base_frame=self.base_frame,
            camera_frame=rospy.get_param('~camera_frame'),
            service_timeout=self.service_timeout,
            tf_timeout=float(rospy.get_param('~tf_timeout', 1.0)))
        self.picker = ManualAcupointPicker(
            color_topic=rospy.get_param('~color_topic'),
            depth_topic=rospy.get_param('~depth_topic'),
            camera_info_topic=rospy.get_param('~camera_info_topic'),
            base_frame=self.base_frame, camera_frame=rospy.get_param('~camera_frame'),
            tf_timeout=float(rospy.get_param('~tf_timeout', 1.0)),
            sync_slop=float(rospy.get_param('~sync_slop', 0.1)),
            image_timeout=float(rospy.get_param('~image_timeout', 1.0)),
            depth_patch_radius=int(rospy.get_param('~depth_patch_radius', 2)))
        self.action = MoxibustionActionClient(self.task['action_name'], self.service_timeout)
        self.workers = {}
        self.motion = 'IDLE'
        self.uncertain = False
        self.external_busy = False
        self.active_goal_ids = set()
        self.terminal_goal_ids = set()
        self.task_open = False
        self.status_received = None
        self.last_feedback_received = None
        self.latest_robot = self.latest_sensor = None
        self.latest_snapshot = self.pick_snapshot = None
        self.base_acupoints = None
        self.detect_ready = False
        self.available = {}
        self.manual_initialized = False
        self.plot_start = time.monotonic()
        self.plot_times, self.plot_temperatures, self.plot_distances, self.plot_h = [], [], [], []
        self.last_h = float('nan')
        self.closing = False
        self._build_ui()
        self.feedback_signal.connect(self._feedback)
        self.done_signal.connect(self._done)
        self.active_signal.connect(self._active)
        self.robot_signal.connect(self._robot_received)
        self.sensor_signal.connect(self._sensor_received)
        self.status_signal.connect(self._status_received)
        self.result_signal.connect(self._result_received)
        self.subscribers = [
            rospy.Subscriber(self.interface_ns + '/state', RobotState,
                             self.robot_signal.emit, queue_size=1),
            rospy.Subscriber(self.task['sensor_topic'], SensorSample,
                             self.sensor_signal.emit, queue_size=1),
            rospy.Subscriber(self.task['action_name'].rstrip('/') + '/status', GoalStatusArray,
                             self.status_signal.emit, queue_size=1),
            rospy.Subscriber(self.task['action_name'].rstrip('/') + '/result',
                             MoxibustionTaskActionResult, self.result_signal.emit, queue_size=10),
        ]
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(200)
        self.readiness_timer = QTimer(self)
        self.readiness_timer.timeout.connect(self._check_readiness)
        self.readiness_timer.start(3000)
        self._check_readiness()
        self._update_controls()

    @staticmethod
    def _spin(low, high, value, decimals=4, step=0.01):
        widget = QDoubleSpinBox()
        widget.setRange(low, high)
        widget.setDecimals(decimals)
        widget.setSingleStep(step)
        widget.setValue(value)
        return widget

    @staticmethod
    def _group(title, parent_layout):
        group = QGroupBox(title)
        layout = QGridLayout(group)
        parent_layout.addWidget(group)
        return group, layout

    def _build_ui(self):
        self.setWindowTitle('艾灸机器人操作台 — PID / SAC-PID')
        self.resize(1520, 940)
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        left_widget = QWidget()
        left = QVBoxLayout(left_widget)
        scroll.setWidget(left_widget)
        main.addWidget(scroll, 7)
        right = QVBoxLayout()
        main.addLayout(right, 5)

        _, layout = self._group('操作', left)
        self.detect_button = QPushButton('穴位检测并转换')
        self.pick_button = QPushButton('冻结图像选点')
        self.live_button = QPushButton('恢复实时图像')
        self.start_button = QPushButton('开始治疗')
        self.cancel_button = QPushButton('取消治疗')
        self.stop_button = QPushButton('请求机器人停止')
        for index, button in enumerate((self.detect_button, self.pick_button, self.live_button,
                                        self.start_button, self.cancel_button, self.stop_button)):
            layout.addWidget(button, index // 3, index % 3)
        self.readiness_label = QLabel('检测准备情况：检查中')
        self.readiness_label.setWordWrap(True)
        layout.addWidget(self.readiness_label, 2, 0, 1, 3)
        self.detect_button.clicked.connect(self._detect)
        self.pick_button.clicked.connect(self._freeze_image)
        self.live_button.clicked.connect(self._live_image)
        self.start_button.clicked.connect(self._start_task)
        self.cancel_button.clicked.connect(self._cancel_task)
        self.stop_button.clicked.connect(self._stop)

        self.detect_group, layout = self._group('穴位检测参数', left)
        self.rotation = QSpinBox()
        self.rotation.setRange(0, 3)
        self.rotation.setValue(int(rospy.get_param('~rotation_k', 3)))
        self.depth_scale = self._spin(0.001, 100.0, float(rospy.get_param('~depth_scale', 1.0)))
        self.cun_unit = self._spin(0.001, 0.200, float(rospy.get_param('~cun_unit', 0.025)))
        for index, (text, widget) in enumerate((('rotation_k', self.rotation),
                ('彩色/深度尺寸比', self.depth_scale), ('同身寸 / m', self.cun_unit))):
            layout.addWidget(QLabel(text), 0, index * 2)
            layout.addWidget(widget, 0, index * 2 + 1)

        self.points_group, layout = self._group('穴位选择与 {} 坐标'.format(self.base_frame), left)
        self.points_combo = QComboBox()
        self.point_label = QLabel('无有效穴位')
        layout.addWidget(self.points_combo, 0, 0)
        layout.addWidget(self.point_label, 1, 0)
        self.points_combo.currentIndexChanged.connect(self._selected_point)

        self.manual_point_group, layout = self._group('手工基座坐标目标 / m', left)
        self.manual_point_spins = []
        for index, axis in enumerate('xyz'):
            spin = self._spin(self.task[axis + '_min_m'], self.task[axis + '_max_m'],
                              self.task['initial_z_m'] if axis == 'z' else 0.0)
            self.manual_point_spins.append(spin)
            layout.addWidget(QLabel(axis), 0, 2 * index)
            layout.addWidget(spin, 0, 2 * index + 1)
        self.manual_target_button = QPushButton('生成手工目标')
        layout.addWidget(self.manual_target_button, 1, 0, 1, 6)
        self.manual_target_button.clicked.connect(self._manual_target)

        self.task_group, layout = self._group('治疗参数（开始前设置）', left)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem('PID', 'pid_control')
        self.mode_combo.addItem('SAC-PID', 'sac_pid_control')
        self.mode_combo.setCurrentIndex(self.mode_combo.findData(self.task['default_mode']))
        self.duration = self._spin(0.0, 600.0, 60.0, 1, 1.0)
        self.duration.setSpecialValueText('0：等待取消')
        self.target_temperature = self._spin(self.task['target_temperature_min_c'],
                                             self.task['target_temperature_max_c'], 43.0, 1, 0.5)
        for index, (text, widget) in enumerate((('方法', self.mode_combo),
                ('时长 / s', self.duration), ('滤波平均温度 / °C', self.target_temperature))):
            layout.addWidget(QLabel(text), 0, 2 * index)
            layout.addWidget(widget, 0, 2 * index + 1)

        self.manual_group, layout = self._group('机械臂手动操作', left)
        self.move_spins = []
        for index, axis in enumerate('xyz'):
            spin = self._spin(self.task[axis + '_min_m'], self.task[axis + '_max_m'], 0.0)
            self.move_spins.append(spin)
            layout.addWidget(QLabel(axis + ' / m'), 0, index * 2)
            layout.addWidget(spin, 0, index * 2 + 1)
        self.angle_spins = [self._spin(-180, 180, 0, 2, 1) for _ in range(3)]
        for index, (axis, spin) in enumerate(zip(('roll', 'pitch', 'yaw'), self.angle_spins)):
            layout.addWidget(QLabel(axis + ' / °'), 1, index * 2)
            layout.addWidget(spin, 1, index * 2 + 1)
        layout.addWidget(QLabel('显示姿态：固定轴 XYZ（sxyz）；位姿接口使用四元数'), 2, 0, 1, 6)
        self.copy_pose_button = QPushButton('从有效反馈填入位姿')
        self.move_button = QPushButton('请求移动')
        self.home_button = QPushButton('Home')
        layout.addWidget(self.copy_pose_button, 3, 0, 1, 2)
        layout.addWidget(self.move_button, 3, 2, 1, 2)
        layout.addWidget(self.home_button, 3, 4, 1, 2)
        self.gripper_position = self._spin(0, 1, 0.5, 3, 0.05)
        self.gripper_button = QPushButton('设置夹爪')
        self.open_button = QPushButton('全开')
        self.close_button = QPushButton('全闭')
        layout.addWidget(QLabel('夹爪目标：0 开 / 1 闭'), 4, 0, 1, 2)
        layout.addWidget(self.gripper_position, 4, 2)
        layout.addWidget(self.gripper_button, 4, 3)
        layout.addWidget(self.open_button, 4, 4)
        layout.addWidget(self.close_button, 4, 5)
        layout.addWidget(QLabel('夹爪实际位置：未知（接口无位置反馈）'), 5, 0, 1, 6)
        self.copy_pose_button.clicked.connect(self._copy_pose)
        self.move_button.clicked.connect(self._move)
        self.home_button.clicked.connect(lambda: self._manual_rpc('home', Trigger))
        self.gripper_button.clicked.connect(lambda: self._gripper(self.gripper_position.value()))
        self.open_button.clicked.connect(lambda: self._gripper(0.0))
        self.close_button.clicked.connect(lambda: self._gripper(1.0))

        _, layout = self._group('实时反馈', left)
        self.value_labels = {}
        for row, (key, label) in enumerate((('temperature', '滤波平均温度 / °C'),
                ('distance', '原始测距 r / m'), ('h', '任务反馈间距 h / m'),
                ('pose', '受控点反馈位姿'), ('target_pose', '任务目标位姿'),
                ('method', '服务器方法与目标'), ('task', '任务状态与结果'))):
            layout.addWidget(QLabel(label), row, 0)
            value = QLabel('未知')
            value.setWordWrap(True)
            layout.addWidget(value, row, 1)
            self.value_labels[key] = value
        _, layout = self._group('反馈信息 / 日志', left)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(500)
        layout.addWidget(self.log, 0, 0)

        self.image_tabs = QTabWidget()
        self.camera_view, self.result_view = ImageView(), ImageView()
        self.image_tabs.addTab(self.camera_view, '相机 / 选点')
        self.image_tabs.addTab(self.result_view, '检测结果')
        self.camera_view.pixel_clicked.connect(self._pick_pixel)
        right.addWidget(self.image_tabs, 3)
        self.temperature_plot = pg.PlotWidget(title='滤波平均温度')
        self.temperature_plot.setLabel('left', '温度', units='°C')
        self.temperature_plot.setLabel('bottom', '界面时间', units='s')
        self.temperature_curve = self.temperature_plot.plot(pen='r', connect='finite')
        right.addWidget(self.temperature_plot, 2)
        self.distance_plot = pg.PlotWidget(title='距离')
        self.distance_plot.setLabel('left', '距离', units='m')
        self.distance_plot.setLabel('bottom', '界面时间', units='s')
        self.distance_plot.addLegend()
        self.distance_curve = self.distance_plot.plot(pen='c', name='原始测距 r', connect='finite')
        self.h_curve = self.distance_plot.plot(pen='y', name='任务反馈 h', connect='finite')
        right.addWidget(self.distance_plot, 2)
        self.statusBar().showMessage('等待机器人与任务服务；初始化不发送运动')

    def _run(self, key, call):
        if key in self.workers:
            return
        worker = CallWorker(key, call)
        self.workers[key] = worker
        worker.result.connect(self._worker_result)
        worker.finished.connect(self._worker_finished)
        worker.start()

    def _worker_finished(self):
        worker = self.sender()
        self.workers.pop(worker.key, None)
        worker.deleteLater()
        self._update_controls()

    def _check_readiness(self):
        if self.closing:
            return
        def probe():
            available = {}
            for name in ('move_to_pose', 'stop', 'home', 'set_gripper_position'):
                if name == 'home' and not self.enable_home or name == 'set_gripper_position' and not self.enable_gripper:
                    available[name] = False
                    continue
                try:
                    rospy.wait_for_service(self.interface_ns + '/' + name, timeout=0.15)
                    available[name] = True
                except rospy.ROSException:
                    available[name] = False
            return available
        self._run('services', probe)
        self._run('readiness', self.workflow.readiness)

    def _worker_result(self, key, result, error):
        if key == 'services':
            self.available = {} if error else result
        elif key == 'readiness':
            self.detect_ready = error is None and bool(result[0])
            self.readiness_label.setText('检测{}：{}'.format(
                '就绪' if self.detect_ready else '不可用', error or result[1]))
        elif key == 'camera':
            if not error:
                self.latest_snapshot = result
                if self.pick_snapshot is None:
                    self.camera_view.show_bgr(result.color_bgr)
            elif self.pick_snapshot is None:
                self.latest_snapshot = None
                self.camera_view.setText('相机不可用：' + error)
        elif key == 'detect':
            if error or not result['success']:
                self._log('检测失败：' + (error or result['message']))
            else:
                self._set_points(result['base_acupoints'])
                self._log(result['message'])
            if not error and result.get('result_image_msg') is not None:
                try:
                    self.result_view.show_bgr(self.bridge.imgmsg_to_cv2(
                        result['result_image_msg'], desired_encoding='bgr8'))
                    self.image_tabs.setCurrentWidget(self.result_view)
                except Exception as exc:
                    self._log('检测图不可用：' + str(exc))
        elif key == 'pick':
            if error or not result['success']:
                self._log('图像选点失败：' + (error or result['message']))
            else:
                self._set_points(result['base_acupoints'])
                self._log('图像选点已转换到 ' + self.base_frame)
                self._live_image()
        elif key == 'send':
            if error:
                self._unknown('发送任务结果未知：' + error)
            elif not result:
                self.task_open = False
                self.motion = 'IDLE'
                self._log('任务已在提交前取消')
            elif self.task_open and self.motion == 'SENDING':
                self.motion = 'PENDING'
        elif key == 'cancel':
            if error:
                self._unknown('取消请求失败：' + error)
            else:
                self._log('取消请求已发送，等待服务器终态')
        elif key == 'stop':
            self._log('停止请求：' + (error or result.message))
            if error or not result.success:
                self._unknown('停止请求未成功；执行状态未知')
        elif key in ('move_to_pose', 'home', 'set_gripper_position'):
            self._log('{}：{}'.format(key, error or result.message))
            if error or not result.success:
                self._unknown('机器人调用未成功；执行状态未知')
            elif not self._valid_robot() or not self._connection_fresh():
                self._unknown('机器人调用返回时通信或反馈无效')
            elif not self.uncertain:
                self.motion = 'IDLE'
        self._update_controls()

    def _log(self, message):
        self.log.append(str(message))

    def _unknown(self, reason):
        self.uncertain = True
        self.motion = 'UNKNOWN'
        self._log(reason + '。运动控件保持锁定；请通过设备操作规程确认状态。')

    def _connection_fresh(self):
        return (self.status_received is not None
                and time.monotonic() - self.status_received <= self.status_timeout)

    def _idle(self):
        pending = any(key in self.workers for key in
                      ('send', 'cancel', 'move_to_pose', 'home', 'set_gripper_position', 'stop'))
        return (not self.closing and not self.uncertain and not self.external_busy
                and self.motion == 'IDLE' and not pending)

    def _update_controls(self):
        idle = self._idle()
        planning = idle and 'detect' not in self.workers and 'pick' not in self.workers
        fresh_pose = self._valid_robot()
        motion_ready = fresh_pose and self._connection_fresh()
        self.detect_button.setEnabled(planning and self.detect_ready)
        self.pick_button.setEnabled(planning and self.latest_snapshot is not None)
        self.live_button.setEnabled(self.pick_snapshot is not None)
        self.start_button.setEnabled(planning and self.base_acupoints is not None
                                     and self._connection_fresh() and fresh_pose)
        self.cancel_button.setEnabled(self.task_open and 'cancel' not in self.workers)
        self.stop_button.setEnabled('stop' not in self.workers and self.available.get('stop', False))
        for widget in (self.detect_group, self.points_group, self.manual_point_group, self.task_group):
            widget.setEnabled(planning)
        self.camera_view.picking = planning and self.pick_snapshot is not None
        self.copy_pose_button.setEnabled(idle and fresh_pose)
        self.move_button.setEnabled(idle and motion_ready and self.manual_initialized
                                    and self.available.get('move_to_pose', False))
        self.home_button.setEnabled(idle and motion_ready and self.enable_home
                                    and self.available.get('home', False))
        gripper = (idle and motion_ready and self.enable_gripper
                   and self.available.get('set_gripper_position', False))
        for widget in (self.gripper_position, self.gripper_button, self.open_button, self.close_button):
            widget.setEnabled(gripper)
        for spin in self.move_spins + self.angle_spins:
            spin.setEnabled(idle)
        self.statusBar().showMessage('{}；Action {}；机器人反馈 {}'.format(
            self.motion, '已连接' if self._connection_fresh() else '未连接/过期',
            '有效' if fresh_pose else '无效/过期'))

    def _status_received(self, message):
        self.status_received = time.monotonic()
        for item in message.status_list:
            goal_id = item.goal_id.id
            if (goal_id and goal_id not in self.terminal_goal_ids
                    and item.status in (GoalStatus.PENDING, GoalStatus.ACTIVE,
                                        GoalStatus.PREEMPTING, GoalStatus.RECALLING)):
                self.active_goal_ids.add(goal_id)
        # 空状态列表或服务器重启不能替代已观察任务的结果。
        self.external_busy = bool(self.active_goal_ids)
        self._update_controls()

    def _result_received(self, message):
        goal_id = message.status.goal_id.id
        if not goal_id:
            return
        tracked = goal_id in self.active_goal_ids
        self.terminal_goal_ids.add(goal_id)
        confirmed = ((message.status.status == GoalStatus.SUCCEEDED and message.result.success)
                     or message.status.status == GoalStatus.PREEMPTED)
        if tracked:
            if confirmed:
                self.active_goal_ids.discard(goal_id)
            else:
                self._unknown('已观察任务的异常结果：' + message.result.message)
        self.external_busy = bool(self.active_goal_ids)
        self._update_controls()

    def _robot_received(self, message):
        self.latest_robot = message

    def _sensor_received(self, message):
        self.latest_sensor = message

    def _fresh(self, stamp):
        age = (rospy.Time.now() - stamp).to_sec()
        return stamp.to_sec() > 0 and 0 <= age <= self.max_age

    def _valid_robot(self):
        state = self.latest_robot
        if state is None or not state.valid or not self._fresh(state.pose.header.stamp):
            return False
        if state.pose.header.frame_id != self.base_frame or not state.controlled_frame:
            return False
        point, quaternion = state.pose.pose.position, state.pose.pose.orientation
        values = (point.x, point.y, point.z, quaternion.x, quaternion.y, quaternion.z, quaternion.w)
        return all(math.isfinite(value) for value in values) and abs(sum(v*v for v in values[3:]) - 1) <= 1e-3

    @staticmethod
    def _value(value):
        return '{:.4f}'.format(value) if math.isfinite(value) else '无有效数据'

    @staticmethod
    def _pose_text(message):
        point, quaternion = message.pose.position, message.pose.orientation
        values = (point.x, point.y, point.z, quaternion.x, quaternion.y, quaternion.z, quaternion.w)
        if not all(math.isfinite(value) for value in values) or abs(sum(v*v for v in values[3:]) - 1) > 1e-3:
            return '无有效位姿'
        angles = tuple(math.degrees(angle) for angle in euler_from_quaternion(values[3:], axes='sxyz'))
        return '{}: XYZ=({:.4f}, {:.4f}, {:.4f}) m; RPY=({:.1f}, {:.1f}, {:.1f})° [sxyz]'.format(
            message.header.frame_id, *values[:3], *angles)

    def _refresh(self):
        if rospy.is_shutdown():
            self._unknown('ROS 已关闭') if not self.uncertain else None
            self._update_controls()
            return
        if (self.task_open or self.active_goal_ids) and not self._connection_fresh():
            self._unknown('任务状态连接中断') if not self.uncertain else None
        manual_pending = any(key in self.workers for key in
                             ('move_to_pose', 'home', 'set_gripper_position'))
        if manual_pending and (not self._valid_robot() or not self._connection_fresh()):
            self._unknown('手动操作期间通信或反馈失效') if not self.uncertain else None
        self._run('camera', self.picker.snapshot)
        sensor = self.latest_sensor
        temperature = distance = float('nan')
        if sensor is not None:
            if sensor.temperature_valid and self._fresh(sensor.temperature_stamp):
                temperature = sensor.temp_filtered_c
            if sensor.distance_valid and self._fresh(sensor.distance_stamp):
                distance = sensor.distance_m
        h = self.last_h if (self.task_open and self.last_feedback_received is not None
                           and time.monotonic() - self.last_feedback_received <= self.max_age) else float('nan')
        self.value_labels['temperature'].setText(self._value(temperature))
        self.value_labels['distance'].setText(self._value(distance))
        self.value_labels['h'].setText(self._value(h))
        state = self.latest_robot
        self.value_labels['pose'].setText(self._pose_text(state.pose) + '；参考点=' + state.controlled_frame
                                          if self._valid_robot() else '无有效位姿')
        for values, value in ((self.plot_times, time.monotonic() - self.plot_start),
                (self.plot_temperatures, temperature), (self.plot_distances, distance), (self.plot_h, h)):
            values.append(value)
            del values[:-1000]
        self.temperature_curve.setData(self.plot_times, self.plot_temperatures)
        self.distance_curve.setData(self.plot_times, self.plot_distances)
        self.h_curve.setData(self.plot_times, self.plot_h)
        self._update_controls()

    def _detect(self):
        if not self._idle():
            return
        values = (self.rotation.value(), self.depth_scale.value(), self.cun_unit.value())
        self._run('detect', lambda: self.workflow.detect_and_transform(*values))
        self._update_controls()

    def _freeze_image(self):
        if not self._idle() or self.latest_snapshot is None:
            return
        self.pick_snapshot = self.latest_snapshot
        self.camera_view.show_bgr(self.pick_snapshot.color_bgr)
        self.image_tabs.setCurrentWidget(self.camera_view)
        self._log('点击冻结图像；深度、内参与变换使用该帧时间')
        self._update_controls()

    def _live_image(self):
        self.pick_snapshot = None
        self._update_controls()

    def _pick_pixel(self, u, v):
        if self.pick_snapshot is None or not self._idle():
            return
        snapshot, scale = self.pick_snapshot, self.depth_scale.value()
        self._run('pick', lambda: self.picker.pick_pixel(u, v, snapshot, 'image_target', scale))
        self._update_controls()

    def _set_points(self, points):
        if points is None or points.header.frame_id != self.base_frame or not points.is_3d:
            self._log('穴位集合的坐标系或三维标记无效')
            return
        valid = [point for point in points.acupoints if point.is_3d and point.confidence >= 0
                 and all(math.isfinite(v) for v in (point.point.x, point.point.y,
                                                     point.point.z, point.confidence))]
        if not valid or len({p.name for p in valid}) != len(valid):
            self._log('穴位集合为空或名称重复')
            return
        self.base_acupoints = copy.deepcopy(points)
        self.base_acupoints.acupoints = valid
        self.points_combo.clear()
        for point in valid:
            self.points_combo.addItem(point.name, point)
        self._selected_point()
        self._update_controls()

    def _selected_point(self, *_):
        point = self.points_combo.currentData()
        self.point_label.setText('无有效穴位' if point is None else
            '{}: ({:.4f}, {:.4f}, {:.4f}) m；置信度 {:.3f}'.format(
                self.base_frame, point.point.x, point.point.y, point.point.z, point.confidence))

    def _manual_target(self):
        if not self._idle():
            return
        point = Acupoint()
        point.name, point.is_3d, point.confidence = 'manual_target', True, 1.0
        point.u = point.v = -1
        point.point = Point(*(spin.value() for spin in self.manual_point_spins))
        points = AcupointArray()
        points.header.frame_id, points.header.stamp = self.base_frame, rospy.Time.now()
        points.is_3d, points.acupoints = True, [point]
        self._set_points(points)

    def _start_task(self):
        if not self._idle() or self.base_acupoints is None or not self._valid_robot():
            return
        point = self.points_combo.currentData()
        mode = self.mode_combo.currentData()
        if point is None or mode not in ('pid_control', 'sac_pid_control'):
            return
        for axis in 'xy':
            if not self.task[axis + '_min_m'] <= getattr(point.point, axis) <= self.task[axis + '_max_m']:
                self._log('目标 ' + axis.upper() + ' 超出任务配置范围')
                return
        temperature, duration = self.target_temperature.value(), self.duration.value()
        if QMessageBox.question(self, '确认开始治疗',
                '{} / {}\n目标 {:.1f} °C，持续 {:.1f} s\n坐标系 {}'.format(
                    point.name, mode, temperature, duration, self.base_frame),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        if not self._idle() or not self._valid_robot() or not self._connection_fresh():
            return
        points, name = copy.deepcopy(self.base_acupoints), point.name
        self.action.prepare()
        self.motion, self.task_open = 'SENDING', True
        self.last_feedback_received, self.last_h = None, float('nan')
        self.value_labels['method'].setText('请求 {} / {:.1f} °C；等待服务器确认'.format(mode, temperature))
        self.value_labels['task'].setText('发送中')
        self._run('send', lambda: self.action.send_goal(points, name, mode, duration, temperature,
                  self.done_signal.emit, self.active_signal.emit, self.feedback_signal.emit))
        self._update_controls()

    def _cancel_task(self):
        if not self.task_open:
            return
        if not self.uncertain:
            self.motion = 'CANCELING'
        self.value_labels['task'].setText('取消中；等待服务器结果')
        self._run('cancel', self.action.cancel_goal)
        self._update_controls()

    def _active(self):
        if self.task_open and self.motion != 'CANCELING' and not self.uncertain:
            self.motion = 'ACTIVE'
        self._update_controls()

    def _feedback(self, feedback):
        if not self.task_open:
            return
        self.last_feedback_received = time.monotonic()
        self.last_h = feedback.current_h
        self.value_labels['method'].setText('{} / {:.1f} °C'.format(
            feedback.mode, feedback.target_temperature))
        self.value_labels['target_pose'].setText(self._pose_text(feedback.target_pose))
        self.value_labels['task'].setText('{:.1f} s | {}'.format(feedback.elapsed.to_sec(), feedback.message))

    def _done(self, status, result):
        self.task_open = False
        names = {GoalStatus.SUCCEEDED: 'SUCCEEDED', GoalStatus.PREEMPTED: 'PREEMPTED',
                 GoalStatus.ABORTED: 'ABORTED', GoalStatus.REJECTED: 'REJECTED',
                 GoalStatus.RECALLED: 'RECALLED', GoalStatus.LOST: 'LOST'}
        detail = '{}；{}'.format(names.get(status, str(status)),
                 '无结果' if result is None else 'success={}；{}'.format(result.success, result.message))
        self.value_labels['task'].setText(detail)
        self._log(detail)
        # PREEMPTED 由任务服务器在停止请求成功后发布，业务 success 仍为 False。
        confirmed = result is not None and ((status == GoalStatus.SUCCEEDED and result.success)
                                             or status == GoalStatus.PREEMPTED)
        if confirmed and not self.uncertain:
            self.motion = 'IDLE'
        else:
            self._unknown('任务异常终态；不能确认执行状态')
        self._update_controls()

    def _copy_pose(self):
        if not self._idle() or not self._valid_robot():
            return
        pose = self.latest_robot.pose.pose
        xyz = (pose.position.x, pose.position.y, pose.position.z)
        if any(not spin.minimum() <= value <= spin.maximum() for spin, value in zip(self.move_spins, xyz)):
            self._log('当前 XYZ 超出界面配置范围，未填入或裁剪位姿')
            return
        q = pose.orientation
        angles = euler_from_quaternion((q.x, q.y, q.z, q.w), axes='sxyz')
        for spin, value in zip(self.move_spins, xyz):
            spin.setValue(value)
        for spin, value in zip(self.angle_spins, angles):
            spin.setValue(math.degrees(value))
        self.manual_initialized = True
        self._update_controls()

    def _move(self):
        if not self._valid_robot() or not self.manual_initialized:
            return
        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.pose.position = Point(*(spin.value() for spin in self.move_spins))
        # 这是界面的固定轴 XYZ 显示约定，厂商旋转协议由后端转换。
        q = quaternion_from_euler(*(math.radians(spin.value()) for spin in self.angle_spins), axes='sxyz')
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = q
        self._manual_rpc('move_to_pose', MoveToPose, pose)

    def _gripper(self, position):
        if self.enable_gripper:
            self._manual_rpc('set_gripper_position', SetGripperPosition, float(position))

    def _call_service(self, name, service_type, *args):
        service = self.interface_ns + '/' + name
        rospy.wait_for_service(service, timeout=self.service_timeout)
        return rospy.ServiceProxy(service, service_type)(*args)

    def _manual_rpc(self, name, service_type, *args):
        if (not self._idle() or not self.available.get(name, False)
                or not self._valid_robot() or not self._connection_fresh()):
            return
        self.motion = 'MANUAL_' + name.upper()
        self._run(name, lambda: self._call_service(name, service_type, *args))
        self._update_controls()

    def _stop(self):
        if self.task_open:
            self._cancel_task()
        self._unknown('已请求机器人停止；响应不证明物理静止')
        self._run('stop', lambda: self._call_service('stop', Trigger))
        self._update_controls()

    def closeEvent(self, event):
        if self.task_open or self.motion.startswith('MANUAL_'):
            QMessageBox.warning(self, '操作未结束', '请先取消任务或请求停止，并等待服务器结果。')
            event.ignore()
            return
        if self.workers:
            self.closing = True
            self.readiness_timer.stop()
            self.timer.stop()
            self._log('后台调用尚未返回，窗口保持打开；可继续使用停止请求。')
            event.ignore()
            return
        self.timer.stop()
        self.readiness_timer.stop()
        for subscriber in self.subscribers:
            subscriber.unregister()
        event.accept()
