"""使用同步 RGB-D 输入生成相机坐标中的穴位。"""

import threading
import time

import cv2
import message_filters
import numpy as np
import rospkg
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger, TriggerResponse
from pathlib import Path

from acupoint_detection.msg import Acupoint, AcupointArray
from acupoint_detection.srv import (
    AcupointDetection, AcupointDetectionResponse, HRNetKeypoints, UNetSegmentation,
)


class AcupointDetectionServer:
    def __init__(self):
        self.bridge = CvBridge()
        self.pipeline = None
        self.pipeline_error = ''
        self.condition = threading.Condition()
        self.detection_lock = threading.Lock()
        self.latest = None
        self.image_timeout = float(rospy.get_param('~image_timeout', 5.0))
        self.max_image_age = float(rospy.get_param('~max_image_age', 1.0))
        self.sync_slop = float(rospy.get_param('~sync_slop', 0.1))
        self.service_timeout = float(rospy.get_param('~service_timeout', 1.0))
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_color_optical_frame')
        self.unet_name = rospy.get_param('~unet_service', 'unet_segmentation')
        self.hrnet_name = rospy.get_param('~hrnet_service', 'hrnet_keypoints')
        self.unet_ready = rospy.get_param('~unet_ready_service', 'unet_segmentation_server/ready')
        self.hrnet_ready = rospy.get_param('~hrnet_ready_service', 'hrnet_keypoints_server/ready')
        self.unet = rospy.ServiceProxy(self.unet_name, UNetSegmentation)
        self.hrnet = rospy.ServiceProxy(self.hrnet_name, HRNetKeypoints)
        self.load_pipeline()
        self.color_sub = message_filters.Subscriber(
            rospy.get_param('~color_topic', '/camera/color/image_raw'), Image)
        self.depth_sub = message_filters.Subscriber(
            rospy.get_param('~depth_topic', '/camera/aligned_depth_to_color/image_raw'), Image)
        self.info_sub = message_filters.Subscriber(
            rospy.get_param('~color_camera_info_topic', '/camera/color/camera_info'), CameraInfo)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub], queue_size=10, slop=self.sync_slop)
        self.sync.registerCallback(self.capture)
        self.ready_service = rospy.Service('~ready', Trigger, self.ready)
        self.detect_service = rospy.Service('~detect_acupoints', AcupointDetection, self.detect)

    def load_pipeline(self):
        try:
            from .pipeline import AcupointDetectionPipeline
            default = Path(rospkg.RosPack().get_path('acupoint_detection')) / 'config/acupoint_info.csv'
            path = Path(rospy.get_param('~acupoints_file', str(default))).expanduser()
            pipeline = AcupointDetectionPipeline()
            pipeline.load_acupoints_info(path)
            if not pipeline.acupoints_info:
                raise ValueError('穴位规则为空')
            self.pipeline = pipeline
        except Exception as exc:
            self.pipeline_error = f'穴位规则不可用: {exc}'
            rospy.logerr(self.pipeline_error)

    def capture(self, color, depth, info):
        with self.condition:
            self.latest = (color, depth, info)
            self.condition.notify_all()

    def validate_snapshot(self, snapshot, depth_scale=None):
        if snapshot is None:
            raise ValueError('尚未收到同步 RGB-D 和 CameraInfo')
        color, depth, info = snapshot
        headers = [item.header for item in snapshot]
        frame = color.header.frame_id
        if not self.camera_frame or frame != self.camera_frame or any(header.frame_id != frame for header in headers):
            raise ValueError('彩色图、对齐深度和 CameraInfo 必须属于同一光学坐标系')
        stamps = [header.stamp.to_sec() for header in headers]
        if any(not np.isfinite(stamp) or stamp <= 0 for stamp in stamps):
            raise ValueError('相机输入时间戳无效')
        if max(stamps) - min(stamps) > self.sync_slop:
            raise ValueError('相机输入时间不同步')
        age = rospy.Time.now().to_sec() - min(stamps)
        if not np.isfinite(age) or age < 0 or age > self.max_image_age:
            raise ValueError('相机输入已过期或时间基准不一致')
        if color.width <= 0 or color.height <= 0 or depth.width <= 0 or depth.height <= 0:
            raise ValueError('图像尺寸无效')
        if (color.width, color.height) != (info.width, info.height):
            raise ValueError('CameraInfo 与彩色图尺寸不一致')
        if depth_scale is not None:
            if abs(depth.width * depth_scale - color.width) > 1 or abs(depth.height * depth_scale - color.height) > 1:
                raise ValueError('深度尺寸与 depth_scale 不一致')
        elif abs(depth.width / color.width - depth.height / color.height) > 1e-6:
            raise ValueError('深度图与彩色图宽高比不一致')
        if depth.encoding not in ('16UC1', 'mono16', '32FC1'):
            raise ValueError('深度编码必须是 16UC1/mono16 毫米或 32FC1 米')
        if len(info.K) != 9 or not np.isfinite(info.K).all() or info.K[0] <= 0 or info.K[4] <= 0:
            raise ValueError('相机内参无效')
        return dict(fx=info.K[0], fy=info.K[4], cx=info.K[2], cy=info.K[5])

    def require_models(self):
        if self.pipeline is None:
            raise RuntimeError(self.pipeline_error)
        for service_name in (self.unet_ready, self.hrnet_ready):
            rospy.wait_for_service(service_name, timeout=self.service_timeout)
            response = rospy.ServiceProxy(service_name, Trigger)()
            if not response.success:
                raise RuntimeError(f'{service_name}: {response.message}')
        for service_name in (self.unet_name, self.hrnet_name):
            rospy.wait_for_service(service_name, timeout=self.service_timeout)

    def ready(self, _request):
        try:
            self.require_models()
            with self.condition:
                snapshot = self.latest
            self.validate_snapshot(snapshot)
            color, depth, _info = snapshot
            rgb = self.bridge.imgmsg_to_cv2(color, 'bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth, 'passthrough')
            if rgb.shape[:2] != (color.height, color.width) or depth_image.shape != (depth.height, depth.width):
                raise ValueError('图像数据与声明尺寸不一致')
            if not np.any(np.isfinite(depth_image) & (depth_image > 0)):
                raise ValueError('深度图没有有效测量值')
            return TriggerResponse(success=True, message='模型、穴位规则与相机输入已就绪')
        except Exception as exc:
            return TriggerResponse(success=False, message=str(exc))

    def wait_snapshot(self):
        deadline = time.monotonic() + self.image_timeout
        with self.condition:
            previous = self.latest[0].header.stamp if self.latest else None
            while not rospy.is_shutdown():
                if self.latest is not None and (previous is None or self.latest[0].header.stamp > previous):
                    return self.latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError('等待新的同步相机输入超时')
                self.condition.wait(min(remaining, 0.1))
        raise RuntimeError('ROS 正在关闭')

    @staticmethod
    def rotate(image, turns):
        if turns == 0:
            return image
        return cv2.rotate(image, (None, cv2.ROTATE_90_CLOCKWISE,
                                 cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE)[turns])

    @staticmethod
    def restore_points(points, turns, height, width):
        if turns == 0:
            return points
        result = {}
        for name, pixel in points.items():
            u, v = np.asarray(pixel, dtype=np.float32)
            if turns == 1:
                u, v = height - 1 - v, u
            elif turns == 2:
                u, v = width - 1 - u, height - 1 - v
            elif turns == 3:
                u, v = v, width - 1 - u
            result[name] = (u, v)
        return result

    @staticmethod
    def project(u, v, depth_mm, intrinsics, depth_scale):
        if not np.isfinite([u, v]).all() or u < 0 or v < 0:
            return None
        u, v = int(u / depth_scale), int(v / depth_scale)
        if not (0 <= u < depth_mm.shape[1] and 0 <= v < depth_mm.shape[0]):
            return None
        z = float(depth_mm[v, u]) * 0.001
        if not np.isfinite(z) or z <= 0:
            return None
        x = (u - intrinsics['cx'] / depth_scale) * z / (intrinsics['fx'] / depth_scale)
        y = (v - intrinsics['cy'] / depth_scale) * z / (intrinsics['fy'] / depth_scale)
        return Point(x=x, y=y, z=z)

    def detect(self, request):
        response = AcupointDetectionResponse()
        started = time.monotonic()
        if not self.detection_lock.acquire(blocking=False):
            response.message = '已有检测请求正在处理'
            return response
        try:
            if request.rotation_k not in (0, 1, 2, 3):
                raise ValueError('rotation_k 必须为 0、1、2 或 3')
            if not np.isfinite([request.depth_scale, request.cun_unit]).all() or request.depth_scale <= 0 or request.cun_unit <= 0:
                raise ValueError('depth_scale 和 cun_unit 必须为正有限数')
            self.require_models()
            color, depth, info = self.wait_snapshot()
            intrinsics = self.validate_snapshot((color, depth, info), request.depth_scale)
            rgb = cv2.cvtColor(self.bridge.imgmsg_to_cv2(color, 'bgr8'), cv2.COLOR_BGR2RGB)
            depth_mm = self.bridge.imgmsg_to_cv2(depth, 'passthrough')
            if depth.encoding == '32FC1':
                depth_mm = depth_mm.astype(np.float64) * 1000.0
            rgb = self.rotate(rgb, request.rotation_k)
            depth_mm = self.rotate(depth_mm, request.rotation_k)
            rotated = self.bridge.cv2_to_imgmsg(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 'bgr8')
            rotated.header = color.header
            unet = self.unet(rotated)
            if not unet.success:
                raise RuntimeError(f'分割不可用: {unet.message}')
            hrnet = self.hrnet(rotated, unet.segmentation_mask)
            if not hrnet.success:
                raise RuntimeError(f'关键点检测不可用: {hrnet.message}')
            mask = self.bridge.imgmsg_to_cv2(unet.segmentation_mask, 'mono8')
            if mask.shape != rgb.shape[:2] or unet.segmentation_mask.header != color.header:
                raise ValueError('分割返回的图像坐标或时间不一致')
            if len(hrnet.keypoints) != 11 or [point.id for point in hrnet.keypoints] != list(range(11)):
                raise ValueError('关键点数量或顺序不符合模型约定')
            if hrnet.header != color.header:
                raise ValueError('关键点返回的图像时间或坐标系不一致')
            points = np.asarray([(point.u, point.v) for point in hrnet.keypoints], dtype=np.float32)
            scores = np.asarray([point.confidence for point in hrnet.keypoints], dtype=np.float32)
            if not np.isfinite(points).all() or not np.isfinite(scores).all() or np.any(scores <= 0):
                raise ValueError('关键点坐标或置信度无效')
            result, acupoints = self.pipeline.process_image(
                rgb, mask, depth_mm, points, scores, request.depth_scale,
                request.cun_unit, intrinsics, plot=True)
            # 反投影使用原始图像方向和原 CameraInfo；像素回映射先于三维计算。
            restore = (4 - request.rotation_k) % 4
            height, width = result.shape[:2]
            acupoints = self.restore_points(acupoints, restore, height, width)
            result = self.rotate(result, restore)
            depth_mm = self.rotate(depth_mm, restore)
            response.acupoints = AcupointArray(header=color.header, is_3d=True)
            rejected = 0
            for name, (u, v) in acupoints.items():
                point = self.project(u, v, depth_mm, intrinsics, request.depth_scale)
                if point is None:
                    rejected += 1
                    continue
                response.acupoints.acupoints.append(Acupoint(
                    name=name, is_3d=True, u=float(u), v=float(v), point=point, confidence=1.0))
            if not response.acupoints.acupoints:
                raise ValueError('未得到具有有效深度的穴位')
            response.result_image = self.bridge.cv2_to_imgmsg(cv2.cvtColor(result, cv2.COLOR_RGB2BGR), 'bgr8')
            response.result_image.header = color.header
            response.success = True
            response.message = f'检测到 {len(response.acupoints.acupoints)} 个有效穴位，排除 {rejected} 个无效深度点'
        except Exception as exc:
            response.message = str(exc)
            rospy.logwarn('穴位检测失败: %s', exc)
        finally:
            response.processing_time = time.monotonic() - started
            self.detection_lock.release()
        return response
