"""轨迹工具模块

本模块提供轨迹采样、导数计算等统一接口，消除分散在多个模块中的重复逻辑。

主要功能:
- 生成轨迹采样时间点
- 计算轨迹在指定时间点的导数
- 批量计算多个时间点的导数

创建日期: 2025-01-06
作者: 代码重构任务
"""

import numpy as np
from typing import Tuple, Optional, Union
from pydrake.trajectories import BsplineTrajectory

from .constants import DEFAULT_NUM_SAMPLES


def generate_sample_times(
    trajectory: BsplineTrajectory,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
) -> np.ndarray:
    """
    生成轨迹采样时间点
    
    统一管理轨迹采样时间点的生成，避免在多个模块中重复使用 np.linspace。
    
    Args:
        trajectory: 轨迹对象（PyDrake BsplineTrajectory）
        num_samples: 采样点数，默认为 DEFAULT_NUM_SAMPLES (100)
        start_time: 起始时间，默认为轨迹起始时间
        end_time: 结束时间，默认为轨迹结束时间
    
    Returns:
        采样时间数组，形状为 (num_samples,)
    
    Examples:
        >>> # 使用默认参数
        >>> t_samples = generate_sample_times(trajectory, num_samples=100)
        >>> 
        >>> # 自定义时间范围
        >>> t_samples = generate_sample_times(trajectory, num_samples=50, 
        ...                                   start_time=0.0, end_time=5.0)
    
    Note:
        此函数替代以下重复代码模式:
        - np.linspace(trajectory.start_time(), trajectory.end_time(), num_samples)
        - np.linspace(start_time, end_time, num_samples)
    """
    t_start = start_time if start_time is not None else trajectory.start_time()
    t_end = end_time if end_time is not None else trajectory.end_time()
    return np.linspace(t_start, t_end, num_samples)


def compute_derivatives(
    trajectory: BsplineTrajectory,
    t: float,
    order: int = 2
) -> Tuple[np.ndarray, ...]:
    """
    计算轨迹在时间t处的导数
    
    统一管理轨迹导数计算，支持任意阶导数。
    
    Args:
        trajectory: 轨迹对象（PyDrake BsplineTrajectory）
        t: 时间点
        order: 导数阶数（1=一阶导数，2=二阶导数，默认为2）
    
    Returns:
        导数元组：(position, first_deriv, second_deriv, ...)
        - position: 位置，形状为 (2,)
        - first_deriv: 一阶导数，形状为 (2,)
        - second_deriv: 二阶导数，形状为 (2,)（如果 order >= 2）
        - ...
    
    Examples:
        >>> # 计算位置和一阶导数
        >>> pos, first = compute_derivatives(trajectory, t, order=1)
        >>> 
        >>> # 计算位置、一阶和二阶导数
        >>> pos, first, second = compute_derivatives(trajectory, t, order=2)
        >>> 
        >>> # 计算高阶导数
        >>> pos, first, second, third = compute_derivatives(trajectory, t, order=3)
    
    Note:
        此函数替代以下重复代码模式:
        - trajectory.value(t)
        - trajectory.EvalDerivative(t, 1)
        - trajectory.EvalDerivative(t, 2)
    """
    # 计算位置
    position = trajectory.value(t).flatten()
    
    # 计算一阶导数
    first_deriv = trajectory.EvalDerivative(t, 1).flatten()
    
    if order == 1:
        return (position, first_deriv)
    
    # 计算二阶导数
    second_deriv = trajectory.EvalDerivative(t, 2).flatten()
    
    if order == 2:
        return (position, first_deriv, second_deriv)
    
    # 计算更高阶导数
    derivatives = [position, first_deriv, second_deriv]
    for i in range(3, order + 1):
        deriv = trajectory.EvalDerivative(t, i).flatten()
        derivatives.append(deriv)
    
    return tuple(derivatives)


def compute_trajectory_derivatives_batch(
    trajectory: BsplineTrajectory,
    t_samples: np.ndarray,
    order: int = 2
) -> dict:
    """
    批量计算轨迹在多个时间点处的导数
    
    对多个时间点进行批量导数计算，返回结构化的结果字典。
    
    Args:
        trajectory: 轨迹对象（PyDrake BsplineTrajectory）
        t_samples: 采样时间数组，形状为 (N,)
        order: 导数阶数（默认为2）
    
    Returns:
        字典，包含：
        - 'position': 位置数组，形状为 (N, 2)
        - 'first_deriv': 一阶导数数组，形状为 (N, 2)
        - 'second_deriv': 二阶导数数组，形状为 (N, 2)（如果 order >= 2）
        - ...
    
    Examples:
        >>> t_samples = generate_sample_times(trajectory, 100)
        >>> derivatives = compute_trajectory_derivatives_batch(trajectory, t_samples)
        >>> positions = derivatives['position']  # (100, 2)
        >>> first_derivs = derivatives['first_deriv']  # (100, 2)
        >>> second_derivs = derivatives['second_deriv']  # (100, 2)
    
    Note:
        此函数用于替代循环中重复调用 trajectory.EvalDerivative 的模式。
    """
    # 初始化存储列表
    positions = []
    first_derivs = []
    second_derivs = []
    higher_derivs = {i: [] for i in range(3, order + 1)}
    
    # 批量计算
    for t in t_samples:
        derivs = compute_derivatives(trajectory, t, order)
        positions.append(derivs[0])
        first_derivs.append(derivs[1])
        
        if order >= 2:
            second_derivs.append(derivs[2])
        
        for i in range(3, order + 1):
            higher_derivs[i].append(derivs[i])
    
    # 构建结果字典
    result = {
        'position': np.array(positions),
        'first_deriv': np.array(first_derivs),
    }
    
    if order >= 2:
        result['second_deriv'] = np.array(second_derivs)
    
    for i in range(3, order + 1):
        result[f'{i}_deriv'] = np.array(higher_derivs[i])
    
    return result


def compute_position_and_derivatives(
    trajectory: BsplineTrajectory,
    t: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算轨迹在时间t处的位置、一阶导数和二阶导数
    
    这是一个便捷函数，专门用于最常见的场景：计算位置、速度和加速度。
    
    Args:
        trajectory: 轨迹对象（PyDrake BsplineTrajectory）
        t: 时间点
    
    Returns:
        元组 (position, first_deriv, second_deriv)：
        - position: 位置，形状为 (2,)
        - first_deriv: 一阶导数（速度），形状为 (2,)
        - second_deriv: 二阶导数（加速度），形状为 (2,)
    
    Examples:
        >>> pos, vel, acc = compute_position_and_derivatives(trajectory, t)
        >>> x, y = pos[0], pos[1]
        >>> x_dot, y_dot = vel[0], vel[1]
        >>> x_ddot, y_ddot = acc[0], acc[1]
    
    Note:
        此函数是 compute_derivatives(t, order=2) 的便捷包装。
    """
    pos, first, second = compute_derivatives(trajectory, t, order=2)
    return pos, first, second


# ============================================================================
# 模块导出
# ============================================================================

__all__ = [
    'generate_sample_times',
    'compute_derivatives',
    'compute_trajectory_derivatives_batch',
    'compute_position_and_derivatives',
]
