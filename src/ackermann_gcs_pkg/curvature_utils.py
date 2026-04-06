"""曲率工具模块

本模块提供曲率计算、曲率梯度计算等统一接口，消除分散在多个模块中的重复逻辑。

主要功能:
- 计算曲率 κ = (ẋÿ - ẏẍ) / (ẋ² + ẏ²)^(3/2)
- 计算曲率对导数的梯度
- 提供曲率计算的详细信息

创建日期: 2025-01-06
作者: 代码重构任务
"""

import numpy as np
from typing import Union, Tuple
from dataclasses import dataclass

from .numerical_safety_utils import safe_power_divide
from .constants import (
    NUMERICAL_TOLERANCE,
    SMALL_VALUE_THRESHOLD,
    CURVATURE_EPSILON,
    CURVATURE_GRADIENT_EPSILON
)


@dataclass
class CurvatureResult:
    """
    曲率计算结果数据类
    
    包含曲率计算的所有相关信息，便于调试和分析。
    
    Attributes:
        curvature: 曲率值 κ (1/m)
        speed: 速度模长 |v| = sqrt(ẋ² + ẏ²) (m/s)
        numerator: 分子 (ẋÿ - ẏẍ) (m²/s³)
        denominator: 分母 (ẋ² + ẏ²)^(3/2) (m³/s³)
    
    Examples:
        >>> result = compute_curvature_with_details(1.0, 0.0, 0.0, 1.0)
        >>> print(f"curvature={result.curvature}, speed={result.speed}")
    """
    curvature: float
    speed: float
    numerator: float
    denominator: float


@dataclass
class CurvatureGradient:
    """
    曲率梯度数据类
    
    包含曲率对各阶导数的偏导数。
    
    Attributes:
        dk_dx_dot: ∂κ/∂ẋ - 曲率对x方向一阶导数的偏导
        dk_dy_dot: ∂κ/∂ẏ - 曲率对y方向一阶导数的偏导
        dk_dx_ddot: ∂κ/∂ẍ - 曲率对x方向二阶导数的偏导
        dk_dy_ddot: ∂κ/∂ÿ - 曲率对y方向二阶导数的偏导
    
    Examples:
        >>> grad = compute_curvature_gradient(1.0, 0.0, 0.0, 1.0)
        >>> print(f"∂κ/∂ẋ={grad.dk_dx_dot}, ∂κ/∂ẏ={grad.dk_dy_dot}")
    """
    dk_dx_dot: float
    dk_dy_dot: float
    dk_dx_ddot: float
    dk_dy_ddot: float


def compute_curvature(
    x_dot: Union[float, np.ndarray],
    y_dot: Union[float, np.ndarray],
    x_ddot: Union[float, np.ndarray],
    y_ddot: Union[float, np.ndarray],
    epsilon: float = SMALL_VALUE_THRESHOLD
) -> Union[float, np.ndarray]:
    """
    计算曲率
    
    曲率公式: κ = (ẋÿ - ẏẍ) / (ẋ² + ẏ²)^(3/2)
    
    此函数统一了分散在8个文件中15处重复的曲率计算逻辑。
    
    Args:
        x_dot: x方向一阶导数 ẋ (m/s)
        y_dot: y方向一阶导数 ẏ (m/s)
        x_ddot: x方向二阶导数 ẍ (m/s²)
        y_ddot: y方向二阶导数 ÿ (m/s²)
        epsilon: 小量保护值，用于避免除零（默认 1e-6）
    
    Returns:
        曲率值 κ (1/m)，标量或数组
    
    Examples:
        >>> # 直线运动（曲率为0）
        >>> kappa = compute_curvature(1.0, 0.0, 0.0, 0.0)
        >>> print(kappa)  # 0.0
        >>> 
        >>> # 圆周运动（曲率为1/R）
        >>> kappa = compute_curvature(0.0, 1.0, -1.0, 0.0)  # R=1
        >>> print(kappa)  # 1.0
        >>> 
        >>> # 数组输入
        >>> x_dot = np.array([1.0, 0.0, 1.0])
        >>> y_dot = np.array([0.0, 1.0, 1.0])
        >>> x_ddot = np.array([0.0, -1.0, 0.0])
        >>> y_ddot = np.array([0.0, 0.0, 1.0])
        >>> kappa = compute_curvature(x_dot, y_dot, x_ddot, y_ddot)
    
    Note:
        替代以下重复代码位置:
        - ackermann_scp_solver.py: 行188, 370
        - flat_output_mapper.py: 行82
        - curvature_statistics.py: 行121
        - curvature_cost_linearizer.py: 行253, 422, 486
        - curvature_peak_cost.py: 行138
        - parallel_curvature_linearizer.py: 行46
        - analytic_gradient_calculator.py: 行242, 429
        - curvature_derivative_cost.py: 行233, 295
    """
    # 计算分子: ẋÿ - ẏẍ
    numerator = x_dot * y_ddot - y_dot * x_ddot
    
    # 计算速度平方: ẋ² + ẏ²
    speed_squared = x_dot**2 + y_dot**2
    
    # 计算速度: sqrt(ẋ² + ẏ²)
    speed = np.sqrt(speed_squared)
    
    # 安全除法: numerator / speed^3
    # 使用 safe_power_divide 避免除零
    return safe_power_divide(numerator, speed, power=3, epsilon=epsilon)


def compute_curvature_with_details(
    x_dot: float,
    y_dot: float,
    x_ddot: float,
    y_ddot: float,
    epsilon: float = SMALL_VALUE_THRESHOLD
) -> CurvatureResult:
    """
    计算曲率并返回详细信息
    
    除了曲率值外，还返回速度、分子、分母等详细信息，便于调试和分析。
    
    Args:
        x_dot: x方向一阶导数 ẋ (m/s)
        y_dot: y方向一阶导数 ẏ (m/s)
        x_ddot: x方向二阶导数 ẍ (m/s²)
        y_ddot: y方向二阶导数 ÿ (m/s²)
        epsilon: 小量保护值（默认 1e-6）
    
    Returns:
        CurvatureResult 对象，包含:
        - curvature: 曲率值
        - speed: 速度模长
        - numerator: 分子
        - denominator: 分母
    
    Examples:
        >>> result = compute_curvature_with_details(1.0, 0.0, 0.0, 1.0)
        >>> print(f"curvature={result.curvature}")
        >>> print(f"speed={result.speed}")
        >>> print(f"numerator={result.numerator}")
        >>> print(f"denominator={result.denominator}")
    """
    # 计算速度
    speed_squared = x_dot**2 + y_dot**2
    speed = np.sqrt(speed_squared)
    
    # 计算分子和分母
    numerator = x_dot * y_ddot - y_dot * x_ddot
    denominator = speed_squared ** 1.5
    
    # 计算曲率
    curvature = safe_power_divide(numerator, speed, power=3, epsilon=epsilon)
    
    return CurvatureResult(
        curvature=curvature,
        speed=speed,
        numerator=numerator,
        denominator=denominator
    )


def compute_curvature_gradient(
    x_dot: float,
    y_dot: float,
    x_ddot: float,
    y_ddot: float,
    epsilon: float = CURVATURE_GRADIENT_EPSILON
) -> CurvatureGradient:
    """
    计算曲率对导数的梯度
    
    梯度公式:
        ∂κ/∂ẋ = (3 * ẋ * ẏ * (ẏ * ẍ - ẋ * ÿ)) / (ẋ² + ẏ²)^(5/2)
        ∂κ/∂ẏ = (3 * ẋ * ẏ * (ẋ * ÿ - ẏ * ẍ)) / (ẋ² + ẏ²)^(5/2)
        ∂κ/∂ẍ = -ẏ / (ẋ² + ẏ²)^(3/2)
        ∂κ/∂ÿ = ẋ / (ẋ² + ẏ²)^(3/2)
    
    此函数统一了分散在2个文件中的梯度计算逻辑。
    
    Args:
        x_dot: x方向一阶导数 ẋ (m/s)
        y_dot: y方向一阶导数 ẏ (m/s)
        x_ddot: x方向二阶导数 ẍ (m/s²)
        y_ddot: y方向二阶导数 ÿ (m/s²)
        epsilon: 小量保护值（默认 1e-10）
    
    Returns:
        CurvatureGradient 对象，包含四个偏导数
    
    Examples:
        >>> grad = compute_curvature_gradient(1.0, 0.0, 0.0, 1.0)
        >>> print(f"∂κ/∂ẋ={grad.dk_dx_dot}")
        >>> print(f"∂κ/∂ẏ={grad.dk_dy_dot}")
        >>> print(f"∂κ/∂ẍ={grad.dk_dx_ddot}")
        >>> print(f"∂κ/∂ÿ={grad.dk_dy_ddot}")
    
    Note:
        替代以下重复代码位置:
        - ackermann_scp_solver.py: 行196-216
        - parallel_curvature_linearizer.py: 行55-68
    """
    # 计算速度平方的各次幂
    speed_sq = x_dot**2 + y_dot**2
    denom_2_5 = speed_sq ** 2.5  # (ẋ² + ẏ²)^(5/2)
    denom_1_5 = speed_sq ** 1.5  # (ẋ² + ẏ²)^(3/2)
    
    # 安全处理小分母
    if denom_2_5 < epsilon:
        denom_2_5 = 1.0
    if denom_1_5 < epsilon:
        denom_1_5 = 1.0
    
    # 计算梯度分量
    # ∂κ/∂ẋ = (3 * ẋ * ẏ * (ẏ * ẍ - ẋ * ÿ)) / (ẋ² + ẏ²)^(5/2)
    dk_dx_dot = (3 * x_dot * y_dot * (y_dot * x_ddot - x_dot * y_ddot)) / denom_2_5
    
    # ∂κ/∂ẏ = (3 * ẋ * ẏ * (ẋ * ÿ - ẏ * ẍ)) / (ẋ² + ẏ²)^(5/2)
    dk_dy_dot = (3 * x_dot * y_dot * (x_dot * y_ddot - y_dot * x_ddot)) / denom_2_5
    
    # ∂κ/∂ẍ = -ẏ / (ẋ² + ẏ²)^(3/2)
    dk_dx_ddot = -y_dot / denom_1_5
    
    # ∂κ/∂ÿ = ẋ / (ẋ² + ẏ²)^(3/2)
    dk_dy_ddot = x_dot / denom_1_5
    
    return CurvatureGradient(
        dk_dx_dot=dk_dx_dot,
        dk_dy_dot=dk_dy_dot,
        dk_dx_ddot=dk_dx_ddot,
        dk_dy_ddot=dk_dy_ddot
    )


def compute_curvature_and_gradient(
    x_dot: float,
    y_dot: float,
    x_ddot: float,
    y_ddot: float,
    epsilon: float = CURVATURE_GRADIENT_EPSILON
) -> Tuple[float, CurvatureGradient]:
    """
    同时计算曲率和梯度（优化版本）
    
    在一次计算中同时得到曲率值和梯度，避免重复计算速度等中间量。
    
    Args:
        x_dot: x方向一阶导数 ẋ (m/s)
        y_dot: y方向一阶导数 ẏ (m/s)
        x_ddot: x方向二阶导数 ẍ (m/s²)
        y_ddot: y方向二阶导数 ÿ (m/s²)
        epsilon: 小量保护值（默认 1e-10）
    
    Returns:
        元组 (curvature, gradient):
        - curvature: 曲率值
        - gradient: CurvatureGradient 对象
    
    Examples:
        >>> kappa, grad = compute_curvature_and_gradient(1.0, 0.0, 0.0, 1.0)
        >>> print(f"curvature={kappa}")
        >>> print(f"∂κ/∂ẋ={grad.dk_dx_dot}")
    """
    # 计算曲率
    curvature = compute_curvature(x_dot, y_dot, x_ddot, y_ddot, epsilon)
    
    # 计算梯度
    gradient = compute_curvature_gradient(x_dot, y_dot, x_ddot, y_ddot, epsilon)
    
    return curvature, gradient


# ============================================================================
# 模块导出
# ============================================================================

__all__ = [
    'CurvatureResult',
    'CurvatureGradient',
    'compute_curvature',
    'compute_curvature_with_details',
    'compute_curvature_gradient',
    'compute_curvature_and_gradient',
]
