"""分割与关键点检测服务。"""

from pathlib import Path
import threading
import time

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from std_srvs.srv import Trigger, TriggerResponse

from acupoint_detection.msg import Keypoint
from acupoint_detection.srv import (
    HRNetKeypoints, HRNetKeypointsResponse,
    UNetSegmentation, UNetSegmentationResponse,
)


class ModelServer:
    def __init__(self):
        self.bridge = CvBridge()
        self.model = None
        self.error = '模型未加载'
        self.lock = threading.Lock()
        self.ready_service = rospy.Service('~ready', Trigger, self.ready)
        try:
            path = Path(rospy.get_param('~weights_path', '')).expanduser()
            if not path.is_file():
                raise ValueError('weights_path 必须指向可读取的部署权重文件')
            self.model = self.load_model(str(path))
            self.error = ''
        except Exception as exc:
            self.error = f'模型不可用: {exc}'
            rospy.logerr(self.error)

    def ready(self, _request):
        return TriggerResponse(success=self.model is not None, message=self.error or '模型已加载')

    def rgb(self, message):
        image = self.bridge.imgmsg_to_cv2(message, 'bgr8')
        if image.size == 0:
            raise ValueError('输入图像为空')
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class UNetServer(ModelServer):
    def __init__(self):
        super().__init__()
        self.service = rospy.Service('unet_segmentation', UNetSegmentation, self.detect)

    def load_model(self, path):
        from .unet_pred import UNetPred
        return UNetPred(path)

    def detect(self, request):
        response = UNetSegmentationResponse()
        response.header = request.rgb_image.header
        started = time.monotonic()
        try:
            if self.model is None:
                raise RuntimeError(self.error)
            image = self.rgb(request.rgb_image)
            with self.lock:
                mask = self.model.get_mask(image)
            if mask.shape != image.shape[:2] or not np.isfinite(mask).all():
                raise ValueError('分割掩膜尺寸或数值无效')
            response.segmentation_mask = self.bridge.cv2_to_imgmsg(mask.astype(np.uint8), 'mono8')
            response.segmentation_mask.header = request.rgb_image.header
            response.success = True
            response.message = '分割完成'
        except Exception as exc:
            response.message = str(exc)
            rospy.logwarn('分割请求失败: %s', exc)
        response.inference_time = time.monotonic() - started
        return response


class HRNetServer(ModelServer):
    KEYPOINT_NAMES = (
        'C7', 'left_acromion', 'right_acromion', 'left_armpit', 'right_armpit',
        'left_ais', 'right_ais', 'left_waist', 'right_waist', 'left_hip', 'right_hip',
    )

    def __init__(self):
        super().__init__()
        self.service = rospy.Service('hrnet_keypoints', HRNetKeypoints, self.detect)

    def load_model(self, path):
        from .hrnet_pred import HRNetPred
        return HRNetPred(path, 'add_mask')

    def detect(self, request):
        response = HRNetKeypointsResponse()
        response.header = request.rgb_image.header
        started = time.monotonic()
        try:
            if self.model is None:
                raise RuntimeError(self.error)
            image = self.rgb(request.rgb_image)
            mask = self.bridge.imgmsg_to_cv2(request.segmentation_mask, 'mono8')
            if mask.shape != image.shape[:2]:
                raise ValueError('分割掩膜与图像尺寸不一致')
            if request.segmentation_mask.header != request.rgb_image.header:
                raise ValueError('分割掩膜与图像不是同一帧')
            with self.lock:
                points, scores = self.model.predict(image, mask)
            if points.shape != (11, 2) or scores.shape != (11,):
                raise ValueError('关键点输出形状不符合 11 点模型')
            if not np.isfinite(points).all() or not np.isfinite(scores).all():
                raise ValueError('关键点输出含非有限值')
            for index, (point, score) in enumerate(zip(points, scores)):
                message = Keypoint()
                message.name = self.KEYPOINT_NAMES[index]
                message.id = index
                message.is_3d = False
                message.u, message.v = map(float, point)
                message.confidence = float(score)
                response.keypoints.append(message)
            response.scores = [float(score) for score in scores]
            response.success = True
            response.message = '关键点检测完成'
        except Exception as exc:
            response.message = str(exc)
            rospy.logwarn('关键点请求失败: %s', exc)
        response.inference_time = time.monotonic() - started
        return response
