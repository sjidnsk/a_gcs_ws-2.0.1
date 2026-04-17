"""
数值安全工具模块

本模块提供数值计算中的安全保护功能，包括：
- 除零保护
- 数值溢出保护
- 边界检查
- 安全除法运算
"""

import numpy as np
from typing import Tuple, Optional, Union

# 默认数值容差 - 从 constants.py 统一导入
from .constants import DEFAULT_EPSILON, DEFAULT_SMALL_VALUE


class NumericalSafetyError(Exception):
    """数值安全异常"""
    pass


def safe_divide(
    numerator: Union[float, np.ndarray],
    denominator: Union[float, np.ndarray],
    epsilon: float = DEFAULT_EPSILON,
    return_zero_on_small_denom: bool = True
) -> Union[float, np.ndarray]:
    """
    安全除法运算，避免除零错误
    
    Args:
        numerator: 分子
        denominator: 分母
        epsilon: 小量保护值
        return_zero_on_small_denom: 当分母过小时是否返回0
    
    Returns:
        安全除法结果
    
    Examples:
        >>> safe_divide(1.0, 0.0)
        0.0
        >>> safe_divide(1.0, 1e-20)
        0.0
        >>> safe_divide(1.0, 2.0)
        0.5
    """
    if isinstance(denominator, np.ndarray):
        # 数组情况
        safe_denom = np.where(
            np.abs(denominator) < epsilon,
            epsilon if not return_zero_on_small_denom else 1.0,
            denominator
        )
        if return_zero_on_small_denom:
            result = np.where(np.abs(denominator) < epsilon, 0.0, numerator / safe_denom)
        else:
            result = numerator / safe_denom
        return result
    else:
        # 标量情况
        if abs(denominator) < epsilon:
            if return_zero_on_small_denom:
                return 0.0
            else:
                return numerator / epsilon
        return numerator / denominator


def safe_power_divide(
    numerator: Union[float, np.ndarray],
    denominator: Union[float, np.ndarray],
    power: float = 3.0,
    epsilon: float = DEFAULT_EPSILON
) -> Union[float, np.ndarray]:
    """
    安全幂次除法运算，避免除零错误
    
    常用于曲率计算: curvature = numerator / speed^power
    
    Args:
        numerator: 分子
        denominator: 分母（底数）
        power: 幂次
        epsilon: 小量保护值
    
    Returns:
        安全除法结果
    
    Examples:
        >>> safe_power_divide(1.0, 0.0, power=3)
        0.0
        >>> safe_power_divide(1.0, 2.0, power=3)
        0.125
    """
    if isinstance(denominator, np.ndarray):
        # 数组情况
        safe_denom = np.where(
            np.abs(denominator) < epsilon,
            epsilon,
            denominator
        )
        # 当分母过小时返回0
        result = np.where(
            np.abs(denominator) < epsilon,
            0.0,
            numerator / (np.abs(safe_denom) ** power)
        )
        return result
    else:
        # 标量情况
        if abs(denominator) < epsilon:
            return 0.0
        return numerator / (abs(denominator) ** power)


def check_tan_safety(
    angle: float,
    threshold: float = np.pi/2 - 1e-6,
    angle_name: str = "angle"
) -> None:
    """
    检查tan函数的安全性，避免在接近π/2时发散
    
    Args:
        angle: 角度值（弧度）
        threshold: 安全阈值，默认为π/2 - 1e-6
        angle_name: 角度名称（用于错误消息）
    
    Raises:
        NumericalSafetyError: 如果角度接近π/2
    
    Examples:
        >>> check_tan_safety(0.5)  # OK
        >>> check_tan_safety(np.pi/2)  # Raises NumericalSafetyError
    """
    if abs(angle) >= threshold:
        raise NumericalSafetyError(
            f"{angle_name} must be < π/2 (got {angle:.6f}, "
            f"threshold={threshold:.6f}). "
            f"tan({angle_name}) would diverge."
        )


def check_positive_value(
    value: float,
    value_name: str,
    allow_zero: bool = False
) -> None:
    """
    检查数值是否为正
    
    Args:
        value: 数值
        value_name: 数值名称（用于错误消息）
        allow_zero: 是否允许为零
    
    Raises:
        NumericalSafetyError: 如果数值不满足条件
    
    Examples:
        >>> check_positive_value(1.0, "wheelbase")  # OK
        >>> check_positive_value(0.0, "wheelbase")  # Raises NumericalSafetyError
        >>> check_positive_value(0.0, "wheelbase", allow_zero=True)  # OK
    """
    if allow_zero:
        if value < 0:
            raise NumericalSafetyError(
                f"{value_name} must be non-negative, got {value}"
            )
    else:
        if value <= 0:
            raise NumericalSafetyError(
                f"{value_name} must be positive, got {value}"
            )


def check_array_bounds(
    array: np.ndarray,
    indices: Tuple[int, ...],
    array_name: str = "array"
) -> bool:
    """
    检查数组索引是否越界
    
    Args:
        array: 数组
        indices: 索引范围 (start, end) 或多维索引
        array_name: 数组名称（用于错误消息）
    
    Returns:
        是否在边界内
    
    Examples:
        >>> arr = np.zeros((100, 100))
        >>> check_array_bounds(arr, (40, 80))  # OK, returns True
        >>> check_array_bounds(arr, (90, 110))  # returns False
    """
    if len(indices) == 2:
        # 一维切片 (start, end)
        start, end = indices
        if start < 0 or end > array.shape[0]:
            return False
        return True
    elif len(indices) == 4:
        # 二维切片 (row_start, row_end, col_start, col_end)
        row_start, row_end, col_start, col_end = indices
        if (row_start < 0 or row_end > array.shape[0] or
            col_start < 0 or col_end > array.shape[1]):
            return False
        return True
    else:
        # 通用多维索引检查
        for i, idx in enumerate(indices):
            if idx < 0 or idx >= array.shape[i]:
                return False
        return True


def safe_array_slice(
    array: np.ndarray,
    row_range: Tuple[int, int],
    col_range: Optional[Tuple[int, int]] = None,
    default_value: float = 0.0
) -> np.ndarray:
    """
    安全数组切片，自动处理越界情况
    
    Args:
        array: 输入数组
        row_range: 行范围 (start, end)
        col_range: 列范围 (start, end)，可选
        default_value: 越界位置的默认值
    
    Returns:
        切片结果
    
    Examples:
        >>> arr = np.zeros((100, 100))
        >>> result = safe_array_slice(arr, (40, 80), (60, 100))
    """
    row_start, row_end = row_range
    row_start = max(0, row_start)
    row_end = min(array.shape[0], row_end)
    
    if col_range is not None:
        col_start, col_end = col_range
        col_start = max(0, col_start)
        col_end = min(array.shape[1], col_end)
        return array[row_start:row_end, col_start:col_end]
    else:
        return array[row_start:row_end]


def safe_norm(
    vector: np.ndarray,
    epsilon: float = DEFAULT_EPSILON
) -> float:
    """
    安全向量范数计算，避免零向量情况
    
    Args:
        vector: 输入向量
        epsilon: 小量保护值
    
    Returns:
        向量范数，至少为epsilon
    
    Examples:
        >>> safe_norm(np.array([0.0, 0.0]))
        1e-10
        >>> safe_norm(np.array([3.0, 4.0]))
        5.0
    """
    norm = np.linalg.norm(vector)
    return max(norm, epsilon)


def check_speed_safety(
    speed: float,
    threshold: float = DEFAULT_SMALL_VALUE,
    return_zero_on_small: bool = True
) -> Tuple[bool, float]:
    """
    检查速度是否安全（避免除零）
    
    Args:
        speed: 速度值
        threshold: 小速度阈值
        return_zero_on_small: 当速度过小时是否返回0
    
    Returns:
        (is_safe, safe_speed): 是否安全, 安全速度值
    
    Examples:
        >>> check_speed_safety(0.0)
        (False, 0.0)
        >>> check_speed_safety(1.0)
        (True, 1.0)
    """
    if speed < threshold:
        if return_zero_on_small:
            return False, 0.0
        else:
            return False, threshold
    return True, speed


def clamp_value(
    value: float,
    min_val: float,
    max_val: float
) -> float:
    """
    将数值限制在指定范围内
    
    Args:
        value: 输入值
        min_val: 最小值
        max_val: 最大值
    
    Returns:
        限制后的值
    
    Examples:
        >>> clamp_value(5.0, 0.0, 10.0)
        5.0
        >>> clamp_value(-1.0, 0.0, 10.0)
        0.0
        >>> clamp_value(15.0, 0.0, 10.0)
        10.0
    """
    return max(min_val, min(max_val, value))


def safe_sqrt(
    value: float,
    epsilon: float = DEFAULT_EPSILON
) -> float:
    """
    安全平方根计算，避免负数输入
    
    Args:
        value: 输入值
        epsilon: 小量保护值
    
    Returns:
        平方根结果
    
    Examples:
        >>> safe_sqrt(4.0)
        2.0
        >>> safe_sqrt(-1.0)
        0.0
    """
    if value < 0:
        if abs(value) < epsilon:
            return 0.0
        else:
            # 警告：负数输入
            return 0.0
    return np.sqrt(value)

