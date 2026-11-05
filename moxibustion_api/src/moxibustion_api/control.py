"""温度外环与距离内环计算；调用方提供同一时间基准的反馈并执行点目标。"""

from dataclasses import dataclass
import math
import time
from typing import Tuple


@dataclass(frozen=True)
class TemperatureSample:
    """参与控制的平均温度 Kalman 值（°C）及其原始接收时刻（s）。"""

    stamp_s: float
    filtered_c: float


@dataclass(frozen=True)
class DistanceSample:
    """已换算的艾条到受热面距离 h（m）及该通道原始接收时刻（s）。"""

    stamp_s: float
    distance_m: float


@dataclass(frozen=True)
class PoseSample:
    """基座系位姿 (x, y, z, qx, qy, qz, qw)，位置为 m，采样时刻为 s。"""

    stamp_s: float
    pose: Tuple[float, ...]


@dataclass(frozen=True)
class TimingLimits:
    """有效温度步长范围、通道年龄上限和配对时间差上限，均以 s 表示。"""

    min_dt_s: float
    max_dt_s: float
    max_age_s: float
    max_skew_s: float


@dataclass(frozen=True)
class OuterStep:
    """外环目标距离（m）及对应温度采样时刻（s）。"""

    d_ref_m: float
    temperature_stamp_s: float


@dataclass(frozen=True)
class InnerStep:
    """待执行 (x, y, z, qx, qy, qz, qw) 和命令 Z 增量（m）；零增量不需移动。"""

    command_pose: Tuple[float, ...]
    commanded_delta_z_m: float


class SampleRejected(ValueError):
    """样本无效或未更新，不提交控制历史；调用方应等待新样本。"""


class ControlFault(RuntimeError):
    """控制计算、时序或高度越界；由任务层处理失败和机器人停止。"""


class TemperatureController:
    """以固定 PID 或 SAC 调整的 PID 增益计算一个目标艾灸距离。"""

    def __init__(self, controller, target_c, distance_limits_m, timing,
                 model_dir=None, pid_gains=(0.005, 0.0025, 0.0),
                 reference_rate_hz=1.0, inference_timeout_s=0.2,
                 local_enter_c=2.0, local_exit_c=2.5,
                 monotonic_clock=time.monotonic):
        """为一次任务初始化历史；仅 sac_pid 模式加载策略。"""
        if controller not in ('pid', 'sac_pid'):
            raise ValueError('仅支持 pid 与 sac_pid')
        if not (reference_rate_hz > 0 and inference_timeout_s > 0
                and 0 < timing.min_dt_s <= timing.max_dt_s
                and timing.max_age_s > 0 and timing.max_skew_s >= 0):
            raise ValueError('控制频率和时间范围必须有效')
        self.controller, self.target_c, self.timing = controller, target_c, timing
        self.local_enter_c, self.local_exit_c = local_enter_c, local_exit_c
        self.d_min, self.d_max = distance_limits_m
        if not all(math.isfinite(x) for x in (self.d_min, self.d_max)) or not 0 < self.d_min < self.d_max:
            raise ValueError('距离范围必须为正且有序')
        self.d_ref_m = None
        self.clock, self.inference_timeout_s = monotonic_clock, inference_timeout_s
        self.previous_dt_s = 1.0 / reference_rate_hz
        self.base_gains = tuple(pid_gains)
        self.gains = self.base_gains
        self.policy = self.config = None
        if controller == 'sac_pid':
            from .policy import SacPolicy
            self.policy = SacPolicy(model_dir)
            self.config = self.policy.config
        self.temperature = self.distance = None
        self.error = self.older_error = self.temperature_rate = 0.0
        self.mode = 'READY'

    def step(self, temperature, distance, now_s, target_c=None):
        """用配对采样推进外环，返回限幅后的目标距离；计算成功后提交历史。"""
        actual = distance.distance_m
        target = self.target_c if target_c is None else target_c
        if actual is None or not all(math.isfinite(x) for x in (temperature.filtered_c, actual, target)):
            raise SampleRejected('控制温度、距离或目标无效')
        temperature_age = now_s - temperature.stamp_s
        distance_age = now_s - distance.stamp_s
        skew = temperature.stamp_s - distance.stamp_s
        if not (0 <= temperature_age <= self.timing.max_age_s
                and 0 <= distance_age <= self.timing.max_age_s
                and 0 <= skew <= self.timing.max_skew_s):
            raise SampleRejected('样本过期、时间倒退或配对时间差过大')
        initial = self.temperature is None
        dt = 0.0 if initial else temperature.stamp_s - self.temperature.stamp_s
        if not initial and dt > self.timing.max_dt_s:
            raise ControlFault('温度采样间隔超过上限')
        if not initial and dt < self.timing.min_dt_s:
            raise SampleRejected('温度未更新或时间间隔过短')
        changed = target != self.target_c
        error = target - temperature.filtered_c
        rate = 0.0 if initial else (temperature.filtered_c - self.temperature.filtered_c) / dt
        acceleration = 0.0 if initial else 2.0 * (rate - self.temperature_rate) / (dt + self.previous_dt_s)
        measured_delta = 0.0 if initial else actual - self.distance.distance_m
        previous_error, older_error = self.error, self.older_error
        mode = self._select_mode(error)
        reset = initial or (self.controller == 'pid' and mode == 'LOCAL'
                            and (self.mode != 'LOCAL' or changed))
        if reset:
            previous_error = older_error = error
        elif changed and self.controller == 'sac_pid':
            # 目标变化只平移误差历史，不改变实测温度导数。
            shift = target - self.target_c
            previous_error += shift
            older_error += shift
        # 从前次限幅参考递推，避免内环死区吞掉累积的小增量。
        anchor = actual if initial or reset else self.d_ref_m
        gains, previous_gains = self.gains, self.gains
        if self.controller == 'sac_pid':
            from .policy import map_gains, normalize_observation
            raw10 = (temperature.filtered_c, target, error, rate, acceleration,
                     actual, measured_delta) + previous_gains
            observation = tuple(float(x) for x in normalize_observation(raw10, self.config))
            if not initial and mode == 'LOCAL':
                started = self.clock()
                action = tuple(float(x) for x in self.policy.predict(observation))
                gains = tuple(float(x) for x in map_gains(action, self.config, self.base_gains))
                elapsed = self.clock() - started
                if (elapsed > self.inference_timeout_s
                        or max(temperature_age, distance_age) + elapsed > self.timing.max_age_s):
                    raise ControlFault('推理超时或输入样本已过期')
        if initial:
            delta = 0.0
        elif mode == 'LOCAL':
            # 正温差对应负距离增量，使艾条靠近受热面。
            kp, ki, kd = gains
            p = -kp * (error - previous_error)
            i = -ki * dt * error
            d = -kd * ((error - previous_error) / dt
                       - (previous_error - older_error) / self.previous_dt_s)
            delta = p + i + d
        else:
            bound = self.d_min if mode == 'HEAT' else self.d_max
            delta = bound - anchor
        raw_reference = anchor + delta
        if not all(math.isfinite(x) for x in (rate, acceleration, raw_reference)):
            raise ControlFault('控制计算产生非有限数')
        reference = min(self.d_max, max(self.d_min, raw_reference))
        result = OuterStep(reference, temperature.stamp_s)
        if initial or self.controller == 'sac_pid' or mode == 'LOCAL':
            self.older_error, self.error = previous_error, error
        self.temperature, self.distance, self.temperature_rate = temperature, distance, rate
        if not initial:
            self.previous_dt_s = dt
        self.target_c, self.gains, self.mode = target, gains, mode
        self.d_ref_m = reference
        return result

    def _select_mode(self, error):
        """按温差选择温区；SAC 的进入和退出阈值形成滞环。"""
        if self.controller == 'pid':
            return ('HEAT' if error > self.local_enter_c else
                    'RETREAT' if error < -self.local_enter_c else 'LOCAL')
        if self.mode == 'LOCAL':
            return ('HEAT' if error >= self.local_exit_c else
                    'RETREAT' if error <= -self.local_exit_c else 'LOCAL')
        return ('HEAT' if error > self.local_enter_c else
                'RETREAT' if error < -self.local_enter_c else 'LOCAL')


class DistanceLoop:
    """追踪外环已有目标，保持固定 XY/姿态，仅生成基座 Z 点目标。"""

    def __init__(self, fixed_xy_m, fixed_orientation, z_limits_m,
                 gain, max_delta_z_m, deadband_m,
                 max_age_s, max_skew_s, max_reference_age_s,
                 positive_z_increases_distance):
        """配置固定 XY/姿态及运动限值；要求基座 +Z 增大艾灸距离。"""
        if not positive_z_increases_distance:
            raise ValueError('需要确认基座 +Z 增大间距')
        values = (*fixed_xy_m, *fixed_orientation, *z_limits_m,
                  gain, max_delta_z_m, deadband_m, max_age_s, max_skew_s,
                  max_reference_age_s)
        if not (all(math.isfinite(x) for x in values)
                and z_limits_m[0] < z_limits_m[1]
                and min(gain, max_delta_z_m, max_age_s, max_reference_age_s) > 0
                and min(deadband_m, max_skew_s) >= 0):
            raise ValueError('距离内环的范围、步长或时效配置无效')
        self.xy, self.orientation = tuple(fixed_xy_m), tuple(fixed_orientation)
        self.z_limits = tuple(z_limits_m)
        self.gain, self.max_delta, self.deadband = gain, max_delta_z_m, deadband_m
        self.max_age, self.max_skew, self.max_reference_age = max_age_s, max_skew_s, max_reference_age_s
        self.reference = self.distance = None

    def update_reference(self, outer_step):
        """接收外环目标及其采样时刻，供后续距离采样追踪。"""
        self.reference = outer_step

    def step(self, distance, feedback, now_s):
        """用新距离和有效位姿生成受限 Z 目标；零增量无需运动。"""
        if self.reference is None:
            raise SampleRejected('尚无目标距离')
        if self.distance is not None and distance.stamp_s <= self.distance.stamp_s:
            raise SampleRejected('距离尚未更新')
        distance_age = now_s - distance.stamp_s
        pose_age = now_s - feedback.stamp_s
        skew = distance.stamp_s - feedback.stamp_s
        reference_age = now_s - self.reference.temperature_stamp_s
        if not (0 <= distance_age <= self.max_age and 0 <= pose_age <= self.max_age
                and abs(skew) <= self.max_skew and 0 <= reference_age <= self.max_reference_age):
            raise SampleRejected('距离、位姿或外环参考已过期，或时间不匹配')
        actual = distance.distance_m
        pose = feedback.pose
        if actual is None or not all(math.isfinite(x) for x in (actual, *pose)):
            raise SampleRejected('距离或位姿无效')
        if not self.z_limits[0] <= pose[2] <= self.z_limits[1]:
            raise ControlFault('实测高度越界')
        error = self.reference.d_ref_m - actual
        deadband = abs(error) <= self.deadband
        raw = 0.0 if deadband else self.gain * error
        if not math.isfinite(raw):
            raise ControlFault('距离内环数值溢出')
        delta = min(self.max_delta, max(-self.max_delta, raw))
        candidate = pose[2] + delta
        z_command = min(self.z_limits[1], max(self.z_limits[0], candidate))
        command = self.xy + (z_command,) + self.orientation
        result = InnerStep(command, z_command - pose[2])
        self.distance = distance
        return result
