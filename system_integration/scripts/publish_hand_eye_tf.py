#!/usr/bin/env python3
"""读取 eye-to-hand 外参并发布基座到相机光学坐标系的静态 TF。"""

import math
from pathlib import Path

import rospy
import tf2_ros
import yaml
from geometry_msgs.msg import TransformStamped
from std_srvs.srv import Trigger, TriggerResponse


class HandEyeTFPublisher:
    def __init__(self):
        self.ready = False
        self.message = "未配置相机外参"
        self.broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.ready_service = rospy.Service("~ready", Trigger, self._readiness)
        base_frame = rospy.get_param("~base_frame", "base_link")
        camera_frame = rospy.get_param("~camera_frame", "camera_color_optical_frame")
        calibration_file = rospy.get_param("~calibration_file", "")
        try:
            if not calibration_file:
                raise ValueError("请通过 calibration_file 指定外参文件")
            if not base_frame or not camera_frame or base_frame == camera_frame:
                raise ValueError("基座与相机坐标系必须明确且不同")
            with Path(calibration_file).expanduser().open(encoding="utf-8") as stream:
                data = yaml.safe_load(stream)
            if data["base_frame"] != base_frame or data["camera_frame"] != camera_frame:
                raise ValueError("外参文件中的 base_frame/camera_frame 与配置不一致")
            translation = data["B_T_C"]["translation"]
            quaternion = data["B_T_C"]["rotation"]["quaternion"]
            xyz = [float(translation[key]) for key in ("x", "y", "z")]
            xyzw = [float(quaternion[key]) for key in ("x", "y", "z", "w")]
            if not all(math.isfinite(value) for value in xyz + xyzw):
                raise ValueError("外参包含非有限值")
            if abs(sum(value * value for value in xyzw) - 1.0) > 1e-3:
                raise ValueError("外参旋转必须是单位四元数")
            transform = TransformStamped()
            transform.header.stamp = rospy.Time.now()
            transform.header.frame_id = base_frame
            transform.child_frame_id = camera_frame
            transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = xyz
            transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w = xyzw
            self.broadcaster.sendTransform(transform)
            self.ready = True
            self.message = "已发布静态 TF: {} -> {}".format(base_frame, camera_frame)
            rospy.loginfo(self.message)
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
            self.message = "相机外参不可用: {}".format(exc)
            rospy.logerr(self.message)

    def _readiness(self, _request):
        return TriggerResponse(success=self.ready, message=self.message)


if __name__ == "__main__":
    rospy.init_node("publish_hand_eye_tf")
    HandEyeTFPublisher()
    rospy.spin()
