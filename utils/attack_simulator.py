"""
utils/attack_simulator.py

用于构造 FDI 攻击和传感器故障。
只修改 target，不修改 feature。
这样便于模拟“训练标签污染”或“负荷测量故障”。
"""

import numpy as np
import torch


def _to_numpy(x):
    """将 torch.Tensor 或 np.ndarray 统一转成 np.ndarray。"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def inject_scale_fdi(target, bus_ids, start, end, scale=1.2):
    """
    乘性 FDI 攻击：
        y_attack = y * scale

    参数：
        target: shape = [N, K]
        bus_ids: 被攻击 bus 列表
        start, end: 攻击时间窗口 [start, end)
        scale: 放大或缩小比例

    返回：
        attacked_target: 被攻击后的 target
        attack_mask: bool 矩阵，标记哪些位置被攻击
    """
    y = _to_numpy(target).copy()
    mask = np.zeros_like(y, dtype=bool)

    y[start:end, bus_ids] = y[start:end, bus_ids] * scale
    mask[start:end, bus_ids] = True

    return y, mask


def inject_bias_fdi(target, bus_ids, start, end, bias=0.1):
    """
    加性 FDI 攻击：
        y_attack = y + bias
    """
    y = _to_numpy(target).copy()
    mask = np.zeros_like(y, dtype=bool)

    y[start:end, bus_ids] = y[start:end, bus_ids] + bias
    mask[start:end, bus_ids] = True

    return y, mask


def inject_ramp_fdi(target, bus_ids, start, end, magnitude=0.2):
    """
    斜坡攻击：
        在 [start, end) 内逐渐增加扰动。
    """
    y = _to_numpy(target).copy()
    mask = np.zeros_like(y, dtype=bool)

    length = end - start
    ramp = np.linspace(0, magnitude, length).reshape(-1, 1)

    y[start:end, bus_ids] = y[start:end, bus_ids] + ramp
    mask[start:end, bus_ids] = True

    return y, mask


def inject_sensor_stuck(target, bus_ids, start, end):
    """
    传感器卡死故障：
        攻击窗口内读数固定为 start 时刻值。
    """
    y = _to_numpy(target).copy()
    mask = np.zeros_like(y, dtype=bool)

    stuck_value = y[start, bus_ids].copy()
    y[start:end, bus_ids] = stuck_value
    mask[start:end, bus_ids] = True

    return y, mask