"""SAC actor 推理、观测归一化和 PID 增益映射。"""

import io
from pathlib import Path
import zipfile

import numpy as np
import yaml


FIELD_NAMES = (
    'T_max_observed', 'T_ref', 'error', 'temperature_rate',
    'temperature_acceleration', 'current_distance', 'previous_distance_increment',
    'previous_applied_Kp', 'previous_applied_Ki', 'previous_applied_Kd',
)
GAIN_NAMES = ('kp', 'ki', 'kd')


def load_model_config(model_dir):
    """读取 config.yaml，校验观测顺序、缩放、裁剪及增益映射。"""
    root = Path(model_dir)
    config = yaml.safe_load((root / 'config.yaml').read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('SAC 部署配置必须是字段映射')
    observation = config.get('observation')
    fields = observation.get('fields') if isinstance(observation, dict) else None
    if not isinstance(fields, list) or len(fields) != len(FIELD_NAMES):
        raise ValueError('SAC observation.fields 必须包含十个观测字段')
    if any(not isinstance(field, dict) or field.get('name') != name
           for field, name in zip(fields, FIELD_NAMES)):
        raise ValueError('观测字段顺序不符')
    keys = ('offset', 'scale', 'clip_low', 'clip_high')
    for field in fields:
        try:
            values = tuple(field[key] for key in keys)
        except KeyError as exc:
            raise ValueError('观测字段 {} 缺少 {}'.format(field['name'], exc.args[0])) from exc
        if (not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values)
                or not np.isfinite(values).all()):
            raise ValueError('观测字段 {} 的缩放和裁剪参数须为有限数'.format(field['name']))
        if field['scale'] == 0 or field['clip_low'] > field['clip_high']:
            raise ValueError('观测字段 {} 的 scale 不得为零，clip 范围须有序'.format(field['name']))
    mapping = config.get('gain_mapping')
    if not isinstance(mapping, dict):
        raise ValueError('SAC 部署配置缺少 gain_mapping')
    for bound in ('lower', 'center', 'upper'):
        if not isinstance(mapping.get(bound), dict) or any(name not in mapping[bound] for name in GAIN_NAMES):
            raise ValueError('gain_mapping.{} 必须包含 kp、ki、kd'.format(bound))
    for name in GAIN_NAMES:
        values = tuple(mapping[bound][name] for bound in ('lower', 'center', 'upper'))
        if (not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values)
                or not np.isfinite(values).all()):
            raise ValueError('增益 {} 的 lower/center/upper 须为有限数'.format(name))
        low, center, high = values
        if center == 0 or not low <= center <= high:
            raise ValueError('增益 {} 的 center 不得为零，且须满足 lower <= center <= upper'.format(name))
    return config


def finite_vector(values, size, label):
    """转换为指定长度的有限 float64 一维向量。"""
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(label + '必须是有限的一维向量，长度为 ' + str(size))
    return vector


def normalize_observation(raw, config):
    """按 FIELD_NAMES 顺序缩放和裁剪观测，返回十维 float32 向量。

    T_max_observed 对应平均温度 Kalman 值，距离及其增量使用实测 h。
    """
    raw = finite_vector(raw, 10, '原始观测')
    result = []
    for value, field in zip(raw, config['observation']['fields']):
        scaled = (value - float(field['offset'])) / float(field['scale'])
        result.append(np.clip(scaled, field['clip_low'], field['clip_high']))
    return np.asarray(result, dtype=np.float32)


def map_gains(action, config, base_gains):
    """将三个 [-1,1] 动作映射为相对任务基准的 Kp/Ki/Kd，不累积调节。"""
    action = finite_vector(action, 3, '策略动作')
    if (np.abs(action) > 1).any():
        raise ValueError('策略动作超出 [-1,1]')
    mapping = config['gain_mapping']
    gains = []
    for value, name, base in zip(action, GAIN_NAMES, base_gains):
        low, center, high = (float(mapping[k][name]) for k in ('lower', 'center', 'upper'))
        if value == -1:
            gain = low
        elif value == 0:
            gain = center
        elif value == 1:
            gain = high
        else:
            gain = center + value * (high - center if value > 0 else center - low)
        gains.append(base * (gain / center))
    return np.asarray(gains, dtype=np.float64)


class SacPolicy:
    """读取 sac_pid.zip 的 actor 权重，在 CPU 上进行确定性推理。"""

    def __init__(self, model_dir):
        """从部署目录载入配置及 policy.pth 中的三层 actor 权重。"""
        self.config = load_model_config(model_dir)
        import torch
        self._torch = torch
        with zipfile.ZipFile(Path(model_dir) / 'sac_pid.zip') as archive:
            weights = torch.load(io.BytesIO(archive.read('policy.pth')),
                                 map_location='cpu', weights_only=True)
        self._layers = []
        for prefix, shape in (('actor.latent_pi.0', (64, 10)),
                              ('actor.latent_pi.2', (64, 64)), ('actor.mu', (3, 64))):
            weight, bias = weights[prefix + '.weight'], weights[prefix + '.bias']
            if tuple(weight.shape) != shape or tuple(bias.shape) != (shape[0],):
                raise ValueError('actor 权重形状不符：' + prefix)
            for tensor in (weight, bias):
                if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                    raise ValueError('actor 权重类型或数值无效：' + prefix)
                tensor.requires_grad_(False)
            self._layers.append((weight, bias))

    def predict(self, observation):
        """将十维归一化观测映射为三个 [-1,1] float32 增益动作。"""
        obs = finite_vector(observation, 10, '归一化观测')
        fields = self.config['observation']['fields']
        if any(v < f['clip_low'] or v > f['clip_high'] for v, f in zip(obs, fields)):
            raise ValueError('观测超出冻结配置范围；请先归一化')
        torch = self._torch
        with torch.inference_mode():
            x = torch.from_numpy(obs.astype(np.float32)).reshape(1, 10)
            for weight, bias in self._layers[:-1]:
                x = torch.relu(torch.nn.functional.linear(x, weight, bias))
            weight, bias = self._layers[-1]
            scaled = torch.tanh(torch.nn.functional.linear(x, weight, bias)).numpy()[0]
        # SB3 的反归一化顺序影响 float32 舍入，不能直接返回 scaled。
        action = -1.0 + (0.5 * (scaled + 1.0) * 2.0)
        if not np.isfinite(action).all():
            raise ValueError('策略产生非有限动作')
        return action
