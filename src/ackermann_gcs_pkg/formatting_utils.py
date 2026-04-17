"""格式化工具模块

本模块提供数据格式化和转换的统一接口，消除分散在多个模块中的重复逻辑。

主要功能:
- 路径长度计算
- 数据格式化

创建日期: 2025-01-06
作者: 代码重构任务
"""

import numpy as np
from typing import Union


def compute_path_length(
    points: np.ndarray
) -> float:
    """
    计算路径长度
    
    计算由一系列点组成的路径的总长度。
    
    Args:
        points: 路径点数组，形状为 (N, 2) 或 (2, N)
            - (N, 2): 每行是一个点的 (x, y) 坐标
            - (2, N): 每列是一个点的 (x, y) 坐标
    
    Returns:
        路径总长度（从第一个点到最后一个点的折线长度）
    
    Examples:
        >>> # 形状 (N, 2)
        >>> points = np.array([[0, 0], [1, 0], [1, 1]])
        >>> length = compute_path_length(points)
        >>> print(length)  # 2.0
        >>> 
        >>> # 形状 (2, N)
        >>> points = np.array([[0, 1, 1], [0, 0, 1]])
        >>> length = compute_path_length(points)
        >>> print(length)  # 2.0
    
    Note:
        替代以下重复代码位置:
        - visualization/ackermann/plot_profiles.py: 行230-233
    """
    # 确保形状为 (N, 2)
    if points.shape[0] == 2 and points.shape[1] > 2:
        points = points.T
    
    # 计算相邻点之间的距离
    diff = np.diff(points, axis=0)
    distances = np.sqrt(np.sum(diff**2, axis=1))
    
    # 累积求和
    return np.sum(distances)


def compute_cumulative_path_length(
    points: np.ndarray
) -> np.ndarray:
    """
    计算累积路径长度
    
    计算从起点到每个点的累积路径长度。
    
    Args:
        points: 路径点数组，形状为 (N, 2) 或 (2, N)
    
    Returns:
        累积路径长度数组，形状为 (N,)
        - 第一个元素为 0（起点）
        - 最后一个元素为总路径长度
    
    Examples:
        >>> points = np.array([[0, 0], [1, 0], [1, 1]])
        >>> lengths = compute_cumulative_path_length(points)
        >>> print(lengths)  # [0.0, 1.0, 2.0]
    
    Note:
        用于绘制以路径长度为横坐标的图表。
    """
    # 确保形状为 (N, 2)
    if points.shape[0] == 2 and points.shape[1] > 2:
        points = points.T
    
    # 计算相邻点之间的距离
    diff = np.diff(points, axis=0)
    distances = np.sqrt(np.sum(diff**2, axis=1))
    
    # 累积求和
    cumulative_lengths = np.zeros(len(points))
    cumulative_lengths[1:] = np.cumsum(distances)
    
    return cumulative_lengths


def normalize_angle_to_pi(
    angle: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    将角度归一化到 [-π, π] 范围
    
    Args:
        angle: 角度（弧度），标量或数组
    
    Returns:
        归一化后的角度，范围 [-π, π]
    
    Examples:
        >>> angle = normalize_angle_to_pi(3 * np.pi)
        >>> print(angle)  # -π
        >>> 
        >>> angle = normalize_angle_to_pi(-3 * np.pi / 2)
        >>> print(angle)  # π/2
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def normalize_angle_to_2pi(
    angle: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    将角度归一化到 [0, 2π) 范围
    
    Args:
        angle: 角度（弧度），标量或数组
    
    Returns:
        归一化后的角度，范围 [0, 2π)
    
    Examples:
        >>> angle = normalize_angle_to_2pi(-np.pi)
        >>> print(angle)  # π
        >>> 
        >>> angle = normalize_angle_to_2pi(3 * np.pi)
        >>> print(angle)  # π
    """
    angle = np.mod(angle, 2 * np.pi)
    return np.where(angle < 0, angle + 2 * np.pi, angle)


def format_value_with_unit(
    value: float,
    unit: str,
    precision: int = 2
) -> str:
    """
    格式化数值并添加单位
    
    Args:
        value: 数值
        unit: 单位字符串
        precision: 小数位数（默认2）
    
    Returns:
        格式化后的字符串
    
    Examples:
        >>> s = format_value_with_unit(1.23456, "m/s")
        >>> print(s)  # "1.23 m/s"
        >>> 
        >>> s = format_value_with_unit(0.00123, "rad", precision=4)
        >>> print(s)  # "0.0012 rad"
    """
    return f"{value:.{precision}f} {unit}"


def format_percentage(
    value: float,
    precision: int = 1
) -> str:
    """
    格式化百分比
    
    Args:
        value: 数值（0-1之间）
        precision: 小数位数（默认1）
    
    Returns:
        格式化后的百分比字符串
    
    Examples:
        >>> s = format_percentage(0.1234)
        >>> print(s)  # "12.3%"
        >>> 
        >>> s = format_percentage(0.9999, precision=2)
        >>> print(s)  # "99.99%"
    """
    return f"{value * 100:.{precision}f}%"


# ============================================================================
# 模块导出
# ============================================================================

__all__ = [
    'compute_path_length',
    'compute_cumulative_path_length',
    'normalize_angle_to_pi',
    'normalize_angle_to_2pi',
    'format_value_with_unit',
    'format_percentage',
]
