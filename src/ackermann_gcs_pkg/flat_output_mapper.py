"""
平坦输出映射器

本模块实现了阿克曼转向车辆的平坦输出映射，将位置轨迹（x, y）映射到车辆状态空间。
"""

import numpy as np
from typing import Dict, Optional
from pydrake.trajectories import BsplineTrajectory
from scipy.optimize import root_scalar

from .ackermann_data_structures import VehicleParams
from .curvature_utils import compute_curvature
from .trajectory_utils import generate_sample_times, compute_derivatives
from .constants import NUMERICAL_TOLERANCE, DEFAULT_NUM_SAMPLES


class FlatOutputMapper:
    """
    平坦输出映射器

    将平坦输出（位置轨迹）映射到车辆状态空间，包括：
    - 速度
    - 航向角
    - 曲率
    - 转向角
    - 加速度
    """

    @staticmethod
    def compute_velocity(x_dot: np.ndarray, y_dot: np.ndarray) -> np.ndarray:
        """
        计算速度

        v = sqrt(x_dot^2 + y_dot^2)

        Args:
            x_dot: x方向速度，形状为(N,)
            y_dot: y方向速度，形状为(N,)

        Returns:
            速度，形状为(N,)
        """
        return np.sqrt(x_dot**2 + y_dot**2)

    @staticmethod
    def compute_heading(x_dot: np.ndarray, y_dot: np.ndarray) -> np.ndarray:
        """
        计算航向角

        theta = arctan2(y_dot, x_dot)

        Args:
            x_dot: x方向速度，形状为(N,)
            y_dot: y方向速度，形状为(N,)

        Returns:
            航向角，形状为(N,)，范围[-π, π]
        """
        return np.arctan2(y_dot, x_dot)

    @staticmethod
    def compute_curvature(
        x_dot: np.ndarray,
        y_dot: np.ndarray,
        x_ddot: np.ndarray,
        y_ddot: np.ndarray,
        epsilon: float = NUMERICAL_TOLERANCE,
    ) -> np.ndarray:
        """
        计算曲率

        κ = (x_dot * y_ddot - y_dot * x_ddot) / (x_dot^2 + y_dot^2)^(3/2)

        Args:
            x_dot: x方向速度，形状为(N,)
            y_dot: y方向速度，形状为(N,)
            x_ddot: x方向加速度，形状为(N,)
            y_ddot: y方向加速度，形状为(N,)
            epsilon: 防止除零的小量

        Returns:
            曲率，形状为(N,)
        """
        # 使用新的工具函数
        return compute_curvature(x_dot, y_dot, x_ddot, y_ddot, epsilon=epsilon)

    @staticmethod
    def compute_steering_angle(curvature: np.ndarray, wheelbase: float) -> np.ndarray:
        """
        计算转向角

        δ = arctan(L * κ)

        Args:
            curvature: 曲率，形状为(N,)
            wheelbase: 车辆轴距（米）

        Returns:
            转向角，形状为(N,)
        """
        return np.arctan(wheelbase * curvature)

    @staticmethod
    def compute_acceleration(
        x_dot: np.ndarray,
        y_dot: np.ndarray,
        x_ddot: np.ndarray,
        y_ddot: np.ndarray,
        epsilon: float = 1e-10,
    ) -> np.ndarray:
        """
        计算加速度

        a = (x_dot * x_ddot + y_dot * y_ddot) / sqrt(x_dot^2 + y_dot^2)

        Args:
            x_dot: x方向速度，形状为(N,)
            y_dot: y方向速度，形状为(N,)
            x_ddot: x方向加速度，形状为(N,)
            y_ddot: y方向加速度，形状为(N,)
            epsilon: 防止除零的小量

        Returns:
            加速度，形状为(N,)
        """
        velocity = np.sqrt(x_dot**2 + y_dot**2)
        # 速度保护
        velocity = np.where(velocity < epsilon, 1.0, velocity)
        numerator = x_dot * x_ddot + y_dot * y_ddot
        return numerator / velocity


def compute_flat_output_mapping(
    trajectory: BsplineTrajectory,
    vehicle_params: VehicleParams,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> Dict[str, np.ndarray]:
    """
    计算平坦输出映射

    将贝塞尔轨迹映射到车辆状态空间，返回位置、速度、航向角、曲率、转向角、加速度。

    Args:
        trajectory: 贝塞尔轨迹对象（BsplineTrajectory）
        vehicle_params: 车辆参数（VehicleParams）
        num_samples: 采样点数，默认为DEFAULT_NUM_SAMPLES

    Returns:
        字典，包含以下键：
        - position: 位置轨迹，形状为(2, N)
        - velocity: 速度轨迹，形状为(N,)
        - heading: 航向角轨迹，形状为(N,)
        - curvature: 曲率轨迹，形状为(N,)
        - steering_angle: 转向角轨迹，形状为(N,)
        - acceleration: 加速度轨迹，形状为(N,)
    """
    # 采样时间点 - 使用新的工具函数
    t_samples = generate_sample_times(trajectory, num_samples)

    # 计算位置、速度、航向角、曲率、转向角、加速度
    position_list = []
    velocity_list = []
    heading_list = []
    curvature_list = []
    steering_angle_list = []
    acceleration_list = []

    for t in t_samples:
        # 计算位置和导数 - 使用新的工具函数
        position, first_deriv, second_deriv = compute_derivatives(trajectory, t, order=2)
        position_list.append(position.reshape(2, 1))

        x_dot = first_deriv[0]
        y_dot = first_deriv[1]
        x_ddot = second_deriv[0]
        y_ddot = second_deriv[1]

        # 计算速度
        velocity = FlatOutputMapper.compute_velocity(x_dot, y_dot)
        velocity_list.append(velocity)

        # 计算航向角
        heading = FlatOutputMapper.compute_heading(x_dot, y_dot)
        heading_list.append(heading)

        # 计算曲率
        curvature = FlatOutputMapper.compute_curvature(x_dot, y_dot, x_ddot, y_ddot)
        curvature_list.append(curvature)

        # 计算转向角
        steering_angle = FlatOutputMapper.compute_steering_angle(curvature, vehicle_params.wheelbase)
        steering_angle_list.append(steering_angle)

        # 计算加速度
        acceleration = FlatOutputMapper.compute_acceleration(x_dot, y_dot, x_ddot, y_ddot)
        acceleration_list.append(acceleration)

    return {
        "position": np.hstack(position_list),  # (2, N)
        "velocity": np.array(velocity_list),
        "heading": np.array(heading_list),
        "curvature": np.array(curvature_list),
        "steering_angle": np.array(steering_angle_list),
        "acceleration": np.array(acceleration_list),
    }
