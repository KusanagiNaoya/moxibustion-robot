"""同步 RGB-D 选点、穴位检测与相机到基座的 TF 转换。"""

import copy
import math
import threading
from dataclasses import dataclass

import message_filters
import numpy as np
import rospy
import tf2_geometry_msgs
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger

from acupoint_detection.msg import Acupoint, AcupointArray
from acupoint_detection.srv import AcupointDetection, AcupointDetectionRequest


def _result():
    return dict(success=False, message="", processing_time=0.0,
                camera_acupoints=None, base_acupoints=None,
                result_image_msg=None, transform_report=None)


class AcupointTransformHelper:
    def __init__(self, base_frame="base_link", camera_frame="camera_color_optical_frame",
                 tf_timeout=3.0):
        if not base_frame or not camera_frame or base_frame == camera_frame:
            raise ValueError("基座与相机坐标系必须明确且不同")
        self.base_frame = base_frame
        self.camera_frame = camera_frame
        self.tf_timeout = tf_timeout
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def lookup(self, stamp):
        transform = self.tf_buffer.lookup_transform(
            self.base_frame, self.camera_frame, stamp, rospy.Duration(self.tf_timeout))
        t, q = transform.transform.translation, transform.transform.rotation
        if not all(math.isfinite(v) for v in (t.x, t.y, t.z, q.x, q.y, q.z, q.w)):
            raise ValueError("外参包含非有限值")
        if abs(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w - 1.0) > 1e-3:
            raise ValueError("外参四元数不是单位四元数")
        return transform

    def transform_acupoint_array(self, acupoints_array):
        report = dict(total=0, transformed=0, skipped=0, message="")
        if acupoints_array is None:
            report["message"] = "没有穴位数据"
            return None, report
        report["total"] = len(acupoints_array.acupoints)
        if acupoints_array.header.frame_id != self.camera_frame:
            report["message"] = "穴位坐标系与配置的相机光学坐标系不一致"
            return None, report
        if acupoints_array.header.stamp.to_sec() <= 0:
            report["message"] = "穴位数据缺少图像时间戳"
            return None, report
        try:
            transform = self.lookup(acupoints_array.header.stamp)
        except (tf2_ros.TransformException, rospy.ROSException, ValueError) as exc:
            report["message"] = "相机到基座的 TF 不可用: {}".format(exc)
            return None, report

        base_array = AcupointArray()
        base_array.header = copy.deepcopy(acupoints_array.header)
        base_array.header.frame_id = self.base_frame
        base_array.is_3d = True
        for acupoint in acupoints_array.acupoints:
            p = acupoint.point
            if (not acupoint.is_3d or acupoint.confidence < 0
                    or not all(math.isfinite(v) for v in (p.x, p.y, p.z, acupoint.confidence))
                    or p.z <= 0):
                report["skipped"] += 1
                continue
            point = PointStamped(header=acupoints_array.header, point=p)
            transformed = tf2_geometry_msgs.do_transform_point(point, transform)
            if not all(math.isfinite(v) for v in (
                    transformed.point.x, transformed.point.y, transformed.point.z)):
                report["skipped"] += 1
                continue
            base_acupoint = copy.deepcopy(acupoint)
            base_acupoint.point = transformed.point
            base_array.acupoints.append(base_acupoint)
            report["transformed"] += 1
        report["message"] = "转换 {} 个点，排除 {} 个无效点".format(
            report["transformed"], report["skipped"])
        return (base_array if base_array.acupoints else None), report


class AcupointWorkflowManager:
    def __init__(self, service_name="/acupoint_detection_server/detect_acupoints",
                 ready_service="/acupoint_detection_server/ready", base_frame="base_link",
                 camera_frame="camera_color_optical_frame", service_timeout=3.0,
                 tf_timeout=3.0):
        self.service_name = service_name
        self.ready_service = ready_service
        self.service_timeout = service_timeout
        self.transform_helper = AcupointTransformHelper(base_frame, camera_frame, tf_timeout)
        self.detect_client = rospy.ServiceProxy(service_name, AcupointDetection)

    def readiness(self):
        """检查模型、图像与 TF；服务等待应由界面工作线程调用。"""
        try:
            rospy.wait_for_service(self.ready_service, timeout=self.service_timeout)
            response = rospy.ServiceProxy(self.ready_service, Trigger)()
            if not response.success:
                return False, response.message
            self.transform_helper.lookup(rospy.Time(0))
            return True, "检测与相机到基座的 TF 可用"
        except (rospy.ROSException, rospy.ServiceException, tf2_ros.TransformException,
                ValueError) as exc:
            return False, "检测不可用: {}".format(exc)

    def detect_and_transform(self, rotation_k=3, depth_scale=1.0, cun_unit=0.025):
        result = _result()
        ready, message = self.readiness()
        if not ready:
            result["message"] = message
            return result
        request = AcupointDetectionRequest()
        request.rotation_k = int(rotation_k)
        request.depth_scale = float(depth_scale)
        request.cun_unit = float(cun_unit)
        try:
            rospy.wait_for_service(self.service_name, timeout=self.service_timeout)
            response = self.detect_client(request)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            result["message"] = "穴位检测调用失败: {}".format(exc)
            return result
        result.update(processing_time=response.processing_time,
                      camera_acupoints=response.acupoints, result_image_msg=response.result_image)
        if not response.success:
            result["message"] = response.message
            return result
        base_array, report = self.transform_helper.transform_acupoint_array(response.acupoints)
        result.update(success=base_array is not None, base_acupoints=base_array,
                      transform_report=report, message=report["message"])
        return result


@dataclass(frozen=True)
class RGBDSnapshot:
    color_bgr: np.ndarray
    depth: np.ndarray
    camera_info: CameraInfo
    header: object
    depth_encoding: str


class ManualAcupointPicker:
    """图像由 Qt 显示；点击必须使用显示时保存的同一份 RGB-D 快照。"""

    def __init__(self, color_topic="/camera/color/image_raw",
                 depth_topic="/camera/aligned_depth_to_color/image_raw",
                 camera_info_topic="/camera/color/camera_info", base_frame="base_link",
                 camera_frame="camera_color_optical_frame", tf_timeout=1.0,
                 sync_slop=0.05, image_timeout=1.0, depth_patch_radius=2):
        self.camera_frame = camera_frame
        self.image_timeout = float(image_timeout)
        self.depth_patch_radius = int(depth_patch_radius)
        self.bridge = CvBridge()
        self.transform_helper = AcupointTransformHelper(base_frame, camera_frame, tf_timeout)
        self._lock = threading.Lock()
        self._snapshot = None
        self._error = "等待同步的彩色图、对齐深度和 CameraInfo"
        self._subscribers = [
            message_filters.Subscriber(color_topic, Image, queue_size=1),
            message_filters.Subscriber(depth_topic, Image, queue_size=1),
            message_filters.Subscriber(camera_info_topic, CameraInfo, queue_size=1)]
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            self._subscribers, queue_size=10, slop=float(sync_slop), allow_headerless=False)
        self._synchronizer.registerCallback(self._on_images)

    def _on_images(self, color_msg, depth_msg, info_msg):
        try:
            messages = (color_msg, depth_msg, info_msg)
            if any(m.header.frame_id != self.camera_frame for m in messages):
                raise ValueError("图像、对齐深度与内参必须使用配置的相机光学坐标系")
            if any(m.header.stamp.to_sec() <= 0 for m in messages):
                raise ValueError("图像或内参时间戳无效")
            if depth_msg.encoding not in ("16UC1", "mono16", "32FC1"):
                raise ValueError("不支持的深度编码: {}".format(depth_msg.encoding))
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8").copy()
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough").copy()
            if color.shape[:2] != (info_msg.height, info_msg.width) or depth.ndim != 2:
                raise ValueError("图像尺寸与 CameraInfo 不匹配，或深度不是单通道")
            if (not all(math.isfinite(k) for k in info_msg.K)
                    or info_msg.K[0] <= 0 or info_msg.K[4] <= 0):
                raise ValueError("相机内参无效")
            color.setflags(write=False)
            depth.setflags(write=False)
            snapshot = RGBDSnapshot(color, depth, copy.deepcopy(info_msg),
                                    copy.deepcopy(color_msg.header), depth_msg.encoding)
            with self._lock:
                self._snapshot, self._error = snapshot, ""
        except Exception as exc:
            with self._lock:
                self._snapshot, self._error = None, str(exc)

    def snapshot(self):
        with self._lock:
            snapshot, error = self._snapshot, self._error
        if snapshot is None:
            raise RuntimeError(error)
        age = (rospy.Time.now() - snapshot.header.stamp).to_sec()
        if not 0 <= age <= self.image_timeout:
            raise RuntimeError("同步图像已过期或时钟不一致")
        return snapshot

    def pick_pixel(self, u, v, snapshot, name="manual_target", depth_scale=1.0):
        result = _result()
        result.update(selected_name=name, camera_point=None, base_point=None,
                      pixel=None, depth_m=None)
        try:
            if not math.isfinite(depth_scale) or depth_scale <= 0:
                raise ValueError("depth_scale 必须为正数")
            u, v = int(u), int(v)
            height, width = snapshot.color_bgr.shape[:2]
            if not (0 <= u < width and 0 <= v < height):
                raise ValueError("点击点超出彩色图像范围")
            depth_height, depth_width = snapshot.depth.shape
            if (not math.isclose(width / depth_width, depth_scale, rel_tol=1e-6)
                    or not math.isclose(height / depth_height, depth_scale, rel_tol=1e-6)):
                raise ValueError("depth_scale 与彩色图和对齐深度的尺寸比例不一致")
            du, dv = int(u / depth_scale), int(v / depth_scale)
            radius = self.depth_patch_radius
            patch = snapshot.depth[max(0, dv-radius):min(depth_height, dv+radius+1),
                                   max(0, du-radius):min(depth_width, du+radius+1)]
            valid = patch[np.isfinite(patch) & (patch > 0)]
            if valid.size == 0:
                raise ValueError("点击点附近没有有效深度")
            # 16UC1/mono16 为毫米，32FC1 为米；depth_scale 只映射像素。
            depth_m = float(np.median(valid)) * (1.0 if snapshot.depth_encoding == "32FC1" else 0.001)
            k = snapshot.camera_info.K
            point = PointStamped()
            point.header = copy.deepcopy(snapshot.header)
            point.point.x = (u - k[2]) * depth_m / k[0]
            point.point.y = (v - k[5]) * depth_m / k[4]
            point.point.z = depth_m
            acupoint = Acupoint(name=name, is_3d=True, u=u, v=v,
                                point=point.point, confidence=1.0)
            camera_array = AcupointArray(header=point.header, is_3d=True, acupoints=[acupoint])
            base_array, report = self.transform_helper.transform_acupoint_array(camera_array)
            result.update(camera_acupoints=camera_array, camera_point=point,
                          pixel=(u, v), depth_m=depth_m, transform_report=report,
                          base_acupoints=base_array, success=base_array is not None,
                          message=report["message"])
            if base_array is not None:
                result["base_point"] = PointStamped(header=base_array.header,
                                                    point=base_array.acupoints[0].point)
        except (ValueError, TypeError, AttributeError) as exc:
            result["message"] = "选点失败: {}".format(exc)
        return result
