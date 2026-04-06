"""约束工具模块

本模块提供约束验证、违反量计算等统一接口，消除分散在多个模块中的重复逻辑。

主要功能:
- 计算约束违反量
- 识别约束违反点
- 检查约束是否满足

创建日期: 2025-01-06
作者: 代码重构任务
"""

import numpy as np
from typing import List, Union
from dataclasses import dataclass

from .constants import (
    VELOCITY_VIOLATION_THRESHOLD,
    ACCELERATION_VIOLATION_THRESHOLD,
    CURVATURE_VIOLATION_THRESHOLD
)


@dataclass
class ViolationResult:
    """
    约束违反结果数据类
    
    包含约束检查的所有相关信息。
    
    Attributes:
        is_violated: 是否违反约束
        max_violation: 最大违反量
        violation_points: 违反点位置列表（归一化到[0,1]）
        violation_indices: 违反点索引列表（可选）
    
    Examples:
        >>> result = check_constraint_satisfaction(1.5, 1.0, threshold=0.01)
        >>> if result.is_violated:
        ...     print(f"最大违反量: {result.max_violation}")
        ...     print(f"违反点: {result.violation_points}")
    """
    is_violated: bool
    max_violation: float
    violation_points: List[float]
    violation_indices: List[int] = None


def compute_constraint_violation(
    value: Union[float, np.ndarray],
    limit: float,
    use_absolute: bool = True
) -> Union[float, np.ndarray]:
    """
    计算约束违反量
    
    违反量定义:
        - 使用绝对值: max(0, |value| - limit)
        - 不使用绝对值: max(0, value - limit)
    
    此函数统一了分散在多个模块中的约束违反量计算逻辑。
    
    Args:
        value: 约束值（标量或数组）
        limit: 约束限制（正数）
        use_absolute: 是否使用绝对值（默认True）
            - True: 适用于双边约束（如加速度、曲率）
            - False: 适用于单边约束（如速度）
    
    Returns:
        违反量（标量或数组），非负数
    
    Examples:
        >>> # 无违反
        >>> violation = compute_constraint_violation(0.5, 1.0)
        >>> print(violation)  # 0.0
        >>> 
        >>> # 有违反
        >>> violation = compute_constraint_violation(1.5, 1.0)
        >>> print(violation)  # 0.5
        >>> 
        >>> # 负值（使用绝对值）
        >>> violation = compute_constraint_violation(-1.5, 1.0)
        >>> print(violation)  # 0.5
        >>> 
        >>> # 数组输入
        >>> values = np.array([0.5, 1.0, 1.5])
        >>> violations = compute_constraint_violation(values, 1.0)
        >>> print(violations)  # [0.0, 0.0, 0.5]
    
    Note:
        替代以下重复代码位置:
        - trajectory_evaluator.py: 行97, 134, 171
        - ackermann_scp_solver.py: 行380
    """
    if use_absolute:
        return np.maximum(0, np.abs(value) - limit)
    else:
        return np.maximum(0, value - limit)


def identify_violation_points(
    violations: np.ndarray,
    threshold: float,
    num_samples: int
) -> tuple[List[float], List[int]]:
    """
    识别约束违反点
    
    找出所有违反量超过阈值的点，并返回其归一化位置和索引。
    
    Args:
        violations: 违反量数组，形状为 (N,)
        threshold: 违反阈值（正数）
        num_samples: 总采样点数
    
    Returns:
        元组 (violation_points, violation_indices):
        - violation_points: 违反点位置列表（归一化到[0,1]）
        - violation_indices: 违反点索引列表
    
    Examples:
        >>> violations = np.array([0.0, 0.01, 0.02, 0.0])
        >>> points, indices = identify_violation_points(violations, threshold=0.005, num_samples=4)
        >>> print(points)  # [0.333..., 0.666...]
        >>> print(indices)  # [1, 2]
    
    Note:
        替代以下重复代码位置:
        - trajectory_evaluator.py: 行100-103, 137-140, 174-177
    """
    violation_points = []
    violation_indices = []
    
    for i, v in enumerate(violations):
        if v > threshold:
            # 归一化位置到 [0, 1]
            position = i / (num_samples - 1) if num_samples > 1 else 0.0
            violation_points.append(position)
            violation_indices.append(i)
    
    return violation_points, violation_indices


def check_constraint_satisfaction(
    value: Union[float, np.ndarray],
    limit: float,
    threshold: float,
    use_absolute: bool = True
) -> ViolationResult:
    """
    检查约束是否满足
    
    综合检查约束满足情况，返回详细的违反信息。
    
    Args:
        value: 约束值（标量或数组）
        limit: 约束限制（正数）
        threshold: 违反阈值（正数）
        use_absolute: 是否使用绝对值（默认True）
    
    Returns:
        ViolationResult 对象，包含:
        - is_violated: 是否违反
        - max_violation: 最大违反量
        - violation_points: 违反点位置
        - violation_indices: 违反点索引
    
    Examples:
        >>> # 标量输入 - 无违反
        >>> result = check_constraint_satisfaction(0.5, 1.0, threshold=0.01)
        >>> print(result.is_violated)  # False
        >>> 
        >>> # 标量输入 - 有违反
        >>> result = check_constraint_satisfaction(1.5, 1.0, threshold=0.01)
        >>> print(result.is_violated)  # True
        >>> print(result.max_violation)  # 0.5
        >>> 
        >>> # 数组输入
        >>> values = np.array([0.5, 1.0, 1.5, 2.0])
        >>> result = check_constraint_satisfaction(values, 1.0, threshold=0.01)
        >>> print(result.is_violated)  # True
        >>> print(result.max_violation)  # 1.0
        >>> print(result.violation_points)  # [0.666..., 1.0]
    """
    # 计算违反量
    violations = compute_constraint_violation(value, limit, use_absolute)
    
    # 计算最大违反量
    if isinstance(violations, np.ndarray):
        max_violation = np.max(violations)
        num_samples = len(violations)
        
        # 识别违反点
        violation_points, violation_indices = identify_violation_points(
            violations, threshold, num_samples
        )
    else:
        # 标量情况
        max_violation = violations
        if max_violation > threshold:
            violation_points = [0.0]
            violation_indices = [0]
        else:
            violation_points = []
            violation_indices = []
    
    return ViolationResult(
        is_violated=max_violation > threshold,
        max_violation=max_violation,
        violation_points=violation_points,
        violation_indices=violation_indices
    )


def compute_max_constraint_violation(
    values: np.ndarray,
    limit: float,
    use_absolute: bool = True
) -> float:
    """
    计算最大约束违反量
    
    便捷函数，只返回最大违反量，不返回详细的违反点信息。
    
    Args:
        values: 约束值数组
        limit: 约束限制
        use_absolute: 是否使用绝对值
    
    Returns:
        最大违反量
    
    Examples:
        >>> values = np.array([0.5, 1.0, 1.5, 2.0])
        >>> max_viol = compute_max_constraint_violation(values, 1.0)
        >>> print(max_viol)  # 1.0
    """
    violations = compute_constraint_violation(values, limit, use_absolute)
    return np.max(violations)


def is_constraint_satisfied(
    value: Union[float, np.ndarray],
    limit: float,
    threshold: float,
    use_absolute: bool = True
) -> bool:
    """
    判断约束是否满足
    
    便捷函数，只返回是否满足，不返回详细信息。
    
    Args:
        value: 约束值（标量或数组）
        limit: 约束限制
        threshold: 违反阈值
        use_absolute: 是否使用绝对值
    
    Returns:
        True 如果约束满足，False 如果违反
    
    Examples:
        >>> # 满足约束
        >>> satisfied = is_constraint_satisfied(0.5, 1.0, threshold=0.01)
        >>> print(satisfied)  # True
        >>> 
        >>> # 违反约束
        >>> satisfied = is_constraint_satisfied(1.5, 1.0, threshold=0.01)
        >>> print(satisfied)  # False
    """
    result = check_constraint_satisfaction(value, limit, threshold, use_absolute)
    return not result.is_violated


# ============================================================================
# 模块导出
# ============================================================================

__all__ = [
    'ViolationResult',
    'compute_constraint_violation',
    'identify_violation_points',
    'check_constraint_satisfaction',
    'compute_max_constraint_violation',
    'is_constraint_satisfied',
]
