#!/usr/bin/env python3
"""采集 SENS 串口帧，发布原始距离与滤波平均温度。"""

import math
import time

import rospy
import serial
import serial.tools.list_ports

from moxibustion_api.msg import SensorSample


class KalmanFilter1D:
    """按实际接收时间间隔累积过程噪声的一维温度滤波器。"""

    def __init__(self, q_per_sec=0.01, r=0.5):
        """q_per_sec 单位为 °C²/s，r 单位为 °C²。"""
        if not all(math.isfinite(x) for x in (q_per_sec, r)) or q_per_sec < 0 or r <= 0:
            raise ValueError('Kalman 噪声参数非法')
        self.q_per_sec, self.r = float(q_per_sec), float(r)
        self.x_est = self.p_est = self.last_time = None

    def update(self, measurement, timestamp):
        """输入平均温度（°C）和递增接收时间（秒），返回滤波温度。"""
        if not all(math.isfinite(x) for x in (measurement, timestamp)) or timestamp <= 0:
            raise ValueError('温度或接收时间非法')
        if self.last_time is None:
            self.x_est, self.p_est = float(measurement), 1.0
        else:
            dt = timestamp - self.last_time
            if dt <= 0:
                raise ValueError('温度接收时间未递增')
            p_pred = self.p_est + self.q_per_sec * dt
            k = p_pred / (p_pred + self.r)
            self.x_est += k * (measurement - self.x_est)
            self.p_est = (1.0 - k) * p_pred
        self.last_time = timestamp
        return self.x_est


class SerialSensorNode:
    def __init__(self):
        self.serial_port = rospy.get_param('~port', 'auto')
        self.baudrate = int(rospy.get_param('~baudrate', 115200))
        self.frame_id = rospy.get_param('~frame_id', 'moxibustion_sensor')
        self.max_frame_bytes = int(rospy.get_param('~max_frame_bytes', 512))
        self.max_pending_bytes = int(rospy.get_param('~max_pending_bytes', 4096))
        self.max_partial_age_s = float(rospy.get_param('~max_partial_age_s', 1.0))
        if min(self.baudrate, self.max_frame_bytes, self.max_pending_bytes) <= 0:
            raise ValueError('波特率和缓冲大小必须为正')
        if not math.isfinite(self.max_partial_age_s) or self.max_partial_age_s <= 0:
            raise ValueError('半帧超时必须为正')
        self.kalman_filter = KalmanFilter1D(
            float(rospy.get_param('~kalman_q_per_sec', 0.01)),
            float(rospy.get_param('~kalman_r', 0.5)))
        self.temp_filtered_c = self.distance_m = float('nan')
        self.temperature_seq = self.distance_seq = 0
        self.temperature_stamp_s = self.distance_stamp_s = 0.0
        self.last_receive_s = None
        self.connected, self.ser = False, None
        self.samples = rospy.Publisher('sensor/sample', SensorSample, queue_size=100)

    @staticmethod
    def parse_sensor_data(data_line):
        """解析完整 SENS 帧：D 为 mm，温度为 °C，DF/TF 为通道有效标志。"""
        parts = data_line.strip().split(',')
        if not parts or parts[0] != 'SENS':
            raise ValueError('不是 SENS 帧')
        fields = {}
        for part in parts[1:]:
            key, sep, value = part.strip().partition('=')
            key = key.upper()
            if not sep or not key or key in fields:
                raise ValueError('字段缺少等号或重复')
            fields[key] = value.strip()
        required = ('D', 'DF', 'T0', 'TMAX', 'TMIN', 'TAVG', 'TF')
        if any(key not in fields for key in required):
            raise ValueError('SENS 缺少必要字段')
        parsed = {key: float(fields[key]) for key in required}
        for flag, keys in (('DF', ('D',)), ('TF', ('T0', 'TMAX', 'TMIN', 'TAVG'))):
            if parsed[flag] not in (0, 1):
                raise ValueError('DF/TF 必须为 0 或 1')
            if parsed[flag] and not all(math.isfinite(parsed[key]) for key in keys):
                raise ValueError('有效通道含非有限数')
        return parsed

    def update_latest(self, parsed, stamp_s):
        """无效通道保留旧值和采样时间；距离只转换为米，不补偿安装偏置。"""
        if not math.isfinite(stamp_s) or stamp_s <= 0:
            raise RuntimeError('ROS 接收时间尚未有效')
        if self.last_receive_s is not None and stamp_s <= self.last_receive_s:
            raise RuntimeError('ROS 接收时间倒退或重复')
        if parsed['TF']:
            self.temp_filtered_c = self.kalman_filter.update(parsed['TAVG'], stamp_s)
            self.temperature_seq += 1
            self.temperature_stamp_s = stamp_s
        if parsed['DF']:
            self.distance_m = parsed['D'] / 1000.0
            self.distance_seq += 1
            self.distance_stamp_s = stamp_s
        self.last_receive_s = stamp_s

    def publish_all(self, stamp_s):
        """valid 表示已连接且存在采样；新鲜度由各通道时间戳判断。"""
        msg = SensorSample()
        msg.header.stamp = rospy.Time.from_sec(stamp_s)
        msg.header.frame_id = self.frame_id
        msg.temperature_valid = self.connected and self.temperature_seq > 0
        msg.temperature_seq = self.temperature_seq
        msg.temperature_stamp = rospy.Time.from_sec(self.temperature_stamp_s)
        msg.temp_filtered_c = self.temp_filtered_c
        msg.distance_valid = self.connected and self.distance_seq > 0
        msg.distance_seq = self.distance_seq
        msg.distance_stamp = rospy.Time.from_sec(self.distance_stamp_s)
        msg.distance_m = self.distance_m
        self.samples.publish(msg)

    def open_serial(self):
        port_name = self.serial_port
        if port_name == 'auto':
            ports = serial.tools.list_ports.comports()
            if not ports:
                raise serial.SerialException('未找到可用串口')
            for port in ports:
                rospy.loginfo('可用串口：%s - %s', port.device, port.description)
            port_name = ports[0].device
            rospy.loginfo('自动选择串口：%s', port_name)
        self.ser = serial.Serial(port_name, self.baudrate, timeout=0.1, exclusive=True)
        self.connected = True
        self.ser.reset_input_buffer()
        rospy.loginfo('串口打开成功：%s，波特率=%d', port_name, self.baudrate)

    def read_and_publish(self):
        partial, partial_started = b'', None
        while not rospy.is_shutdown():
            pending_bytes = self.ser.in_waiting
            if pending_bytes > self.max_pending_bytes:
                raise RuntimeError('串口积压超过阈值，需重新检查采集节拍')
            chunk = self.ser.read_until(b'\n', self.max_frame_bytes + 1)
            stamp_s = rospy.Time.now().to_sec()
            now = time.monotonic()
            if chunk and partial_started is None:
                partial_started = now
            partial += chunk
            if len(partial) > self.max_frame_bytes:
                raise RuntimeError('串口帧超过长度限制')
            if partial_started is not None and now - partial_started > self.max_partial_age_s:
                raise RuntimeError('串口半帧超时')
            if not partial.endswith(b'\n'):
                self.publish_all(stamp_s)
                continue
            data, partial, partial_started = partial, b'', None
            line = data.decode('ascii', errors='replace').strip()
            try:
                parsed = self.parse_sensor_data(line)
            except ValueError as exc:
                self.publish_all(stamp_s)
                rospy.logwarn_throttle(5.0, '无效 SENS 帧：%s', str(exc))
                continue
            self.update_latest(parsed, stamp_s)
            self.publish_all(stamp_s)

    def cleanup(self):
        self.connected = False
        if self.ser is not None:
            self.ser.close()
        if not rospy.is_shutdown():
            self.publish_all(rospy.Time.now().to_sec())

    def run(self):
        try:
            self.open_serial()
            self.read_and_publish()
        except (serial.SerialException, OSError, RuntimeError, ValueError) as exc:
            self.connected = False
            self.publish_all(rospy.Time.now().to_sec())
            rospy.logerr('采集已停止：%s', str(exc))
            raise
        finally:
            self.cleanup()


def main():
    rospy.init_node('serial_sensor_node')
    SerialSensorNode().run()


if __name__ == '__main__':
    main()
