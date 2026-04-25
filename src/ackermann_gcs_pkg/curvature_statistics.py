"""
曲率统计模块

本模块实现曲率统计计算和平滑度评估功能，包括：
- 曲率峰值统计
- 平滑度改善量化
- 曲率分布获取
"""

import numpy as np
from typing import Tuple

from .ackermann_data_structures import (
    CurvatureStats,
    ImprovementMetrics
)

# 导入新的工具模块
from .constants import DEFAULT_NUM_SAMPLES, SMALL_VALUE_THRESHOLD
from .curvature_utils import compute_curvature


class CurvatureStatistics:
    """
    曲率统计模块

    提供曲率统计计算和平滑度评估功能。
    """

    def __init__(self, num_samples: int = DEFAULT_NUM_SAMPLES):
        """
        初始化曲率统计模块

        Args:
            num_samples: 采样点数
        """
        self.num_samples = num_samples

    def _compute_trajectory_derivatives(
        self,
        trajectory,
        t: float,
        order: int = 2
    ) -> Tuple[np.ndarray, ...]:
        """
        计算轨迹在时间t处的导数

        Args:
            trajectory: 轨迹对象（BsplineTrajectory）
            t: 时间参数值
            order: 导数阶数

        Returns:
            导数元组
        """
        # 优先使用EvalDerivative（与flat_output_mapper一致）
        if hasattr(trajectory, 'EvalDerivative'):
            first_deriv = trajectory.EvalDerivative(t, 1).flatten()
            if order >= 2:
                second_deriv = trajectory.EvalDerivative(t, 2).flatten()
                return first_deriv, second_deriv
            return (first_deriv,)
        elif hasattr(trajectory, 'derivative'):
            first_deriv = trajectory.derivative(1).value(t)
            if order >= 2:
                second_deriv = trajectory.derivative(2).value(t)
                return first_deriv, second_deriv
            return (first_deriv,)
        elif hasattr(trajectory, 'CalcValue') and hasattr(trajectory, 'CalcDerivative'):
            # Drake风格的API
            first_deriv = trajectory.CalcDerivative(t, 1)
            if order >= 2:
                second_deriv = trajectory.CalcDerivative(t, 2)
                return first_deriv, second_deriv
            return (first_deriv,)
        else:
            # 使用数值差分
            epsilon = 1e-6
            if hasattr(trajectory, 'value'):
                value_method = trajectory.value
            elif hasattr(trajectory, 'CalcValue'):
                value_method = trajectory.CalcValue
            else:
                raise ValueError("Trajectory object has no value method")

            # 一阶导数
            first_deriv = (value_method(t + epsilon) - value_method(t - epsilon)) / (2 * epsilon)

            if order >= 2:
                # 二阶导数
                second_deriv = (value_method(t + epsilon) - 2 * value_method(t) + value_method(t - epsilon)) / (epsilon**2)
                return first_deriv, second_deriv

            return (first_deriv,)

    def _compute_curvature_at_point(
        self,
        trajectory,
        t: float
    ) -> float:
        """
        计算轨迹在时间t处的曲率

        曲率公式：κ = (ẋÿ - ẏẍ) / (ẋ² + ẏ²)^(3/2)

        Args:
            trajectory: 轨迹对象
            t: 时间参数值

        Returns:
            曲率值
        """
        # 计算一阶和二阶导数
        first_deriv, second_deriv = self._compute_trajectory_derivatives(
            trajectory, t, order=2
        )

        x_dot = first_deriv[0]
        y_dot = first_deriv[1]
        x_ddot = second_deriv[0]
        y_ddot = second_deriv[1]

        # 使用新的工具函数计算曲率
        return compute_curvature(x_dot, y_dot, x_ddot, y_ddot, epsilon=SMALL_VALUE_THRESHOLD)

    def compute_curvature_stats(
        self,
        trajectory
    ) -> CurvatureStats:
        """
        计算轨迹的曲率统计信息

        采样轨迹，计算每个采样点的曲率，统计：
        - max_curvature: 最大曲率
        - min_curvature: 最小曲率
        - mean_curvature: 平均曲率
        - std_curvature: 曲率标准差
        - max_curvature_location: 最大曲率位置

        Args:
            trajectory: 轨迹对象（BsplineTrajectory）

        Returns:
            曲率统计信息
        """
        # 采样轨迹时间范围 [start_time, end_time]
        t_start = trajectory.start_time()
        t_end = trajectory.end_time()
        t_array = np.linspace(t_start, t_end, self.num_samples)

        # 计算每个采样点的曲率
        kappa_array = np.array([
            self._compute_curvature_at_point(trajectory, t)
            for t in t_array
        ])

        # 统计计算（所有统计量统一在 |κ| 上计算，保持语义一致性）
        abs_kappa = np.abs(kappa_array)
        max_kappa = np.max(abs_kappa)
        min_kappa = np.min(abs_kappa)
        mean_kappa = np.mean(abs_kappa)
        std_kappa = np.std(abs_kappa)

        # 找到最大曲率位置
        max_idx = np.argmax(np.abs(kappa_array))
        max_curvature_location = t_array[max_idx]

        return CurvatureStats(
            max_curvature=max_kappa,
            min_curvature=min_kappa,
            mean_curvature=mean_kappa,
            std_curvature=std_kappa,
            max_curvature_location=max_curvature_location,
            curvature_samples=kappa_array
        )

    def compare_with_baseline(
        self,
        optimized: CurvatureStats,
        baseline: CurvatureStats
    ) -> ImprovementMetrics:
        """
        与基准轨迹对比，量化平滑度改善

        计算改善百分比：
        - peak_reduction = (baseline.max - optimized.max) / baseline.max * 100%
        - std_reduction = (baseline.std - optimized.std) / baseline.std * 100%
        - mean_reduction = (baseline.mean - optimized.mean) / baseline.mean * 100%

        Args:
            optimized: 优化后的曲率统计
            baseline: 基准曲率统计

        Returns:
            改善指标
        """
        # 计算峰值降低百分比
        if baseline.max_curvature > 0:
            peak_reduction = (
                (baseline.max_curvature - optimized.max_curvature) /
                baseline.max_curvature * 100
            )
        else:
            peak_reduction = 0.0

        # 计算标准差降低百分比
        if baseline.std_curvature > 0:
            std_reduction = (
                (baseline.std_curvature - optimized.std_curvature) /
                baseline.std_curvature * 100
            )
        else:
            std_reduction = 0.0

        # 计算平均值降低百分比
        if baseline.mean_curvature > 0:
            mean_reduction = (
                (baseline.mean_curvature - optimized.mean_curvature) /
                baseline.mean_curvature * 100
            )
        else:
            mean_reduction = 0.0

        return ImprovementMetrics(
            peak_reduction_percent=peak_reduction,
            std_reduction_percent=std_reduction,
            mean_reduction_percent=mean_reduction,
            baseline_stats=baseline,
            optimized_stats=optimized
        )

    def get_curvature_distribution(
        self,
        trajectory,
        num_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取曲率沿轨迹的分布

        Args:
            trajectory: 轨迹对象（BsplineTrajectory）
            num_samples: 采样点数

        Returns:
            (s_array, kappa_array): 参数数组和曲率数组
        """
        s_array = np.linspace(trajectory.start_time(), trajectory.end_time(), num_samples)
        kappa_array = np.array([
            self._compute_curvature_at_point(trajectory, t)
            for t in s_array
        ])
        return s_array, kappa_array
