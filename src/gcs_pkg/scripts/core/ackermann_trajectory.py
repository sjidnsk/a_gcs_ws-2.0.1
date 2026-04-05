"""
阿克曼轨迹封装模块

提供AckermannTrajectory类，封装阿克曼车辆的轨迹数据，
提供基于实际时间的轨迹查询接口。
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from scipy.optimize import root_scalar

from pydrake.trajectories import (
    BsplineTrajectory,
    BsplineTrajectory_,
)


__all__ = ['AckermannTrajectory']


class AckermannTrajectory:
    """
    阿克曼车辆轨迹封装类

    提供基于实际时间的轨迹查询接口，包括：
    - 位置查询：[x(t), y(t), θ(t)]
    - 状态查询：[x(t), y(t), θ(t), v(t), δ(t)]
    - 控制查询：[a(t), ω(t)]
    """

    def __init__(
        self,
        flat_trajectory: BsplineTrajectory,
        time_trajectory: BsplineTrajectory,
        wheelbase: float
    ):
        """
        初始化阿克曼轨迹

        Args:
            flat_trajectory: 平坦输出轨迹 [x, y, θ]
            time_trajectory: 时间轨迹 h(s)
            wheelbase: 车辆轴距
        """
        self.flat_trajectory = flat_trajectory
        self.time_trajectory = time_trajectory
        self.wheelbase = wheelbase

        self.start_s = flat_trajectory.start_time()
        self.end_s = flat_trajectory.end_time()

    def invert_time_traj(self, t: float) -> float:
        """
        将实际时间t映射到参数s

        Args:
            t: 实际时间

        Returns:
            参数s
        """
        if t <= self.start_time():
            return self.start_s
        if t >= self.end_time():
            return self.end_s

        error = lambda s: self.time_trajectory.value(s)[0, 0] - t
        res = root_scalar(
            error,
            bracket=[self.start_s, self.end_s],
            method='brentq'
        )
        return res.root

    def value(self, t: float) -> np.ndarray:
        """
        获取t时刻的位置

        Args:
            t: 实际时间

        Returns:
            位置 [x, y, θ]
        """
        s = self.invert_time_traj(t)
        return self.flat_trajectory.value(s).flatten()

    def EvalDerivative(self, t: float, derivative_order: int = 1) -> np.ndarray:
        """
        获取t时刻的导数

        Args:
            t: 实际时间
            derivative_order: 导数阶数（1=速度，2=加速度）

        Returns:
            导数
        """
        s = self.invert_time_traj(t)
        s_dot = 1.0 / self.time_trajectory.EvalDerivative(s, 1)[0, 0]

        if derivative_order == 1:
            flat_deriv = self.flat_trajectory.EvalDerivative(s, 1)
            return flat_deriv.flatten() * s_dot
        elif derivative_order == 2:
            flat_deriv1 = self.flat_trajectory.EvalDerivative(s, 1)
            flat_deriv2 = self.flat_trajectory.EvalDerivative(s, 2)
            time_ddot = self.time_trajectory.EvalDerivative(s, 2)[0, 0]
            s_ddot = -time_ddot * s_dot**3

            return (
                flat_deriv2.flatten() * s_dot**2 +
                flat_deriv1.flatten() * s_ddot
            )
        else:
            raise ValueError("Unsupported derivative order")

    def get_state(self, t: float) -> np.ndarray:
        """
        获取t时刻的完整状态

        Args:
            t: 实际时间

        Returns:
            状态 [x, y, θ, v, δ]
        """
        flat_output = self.value(t)
        flat_derivs = [
            self.EvalDerivative(t, i)
            for i in range(1, 3)
        ]

        # 运行时导入，避免循环依赖
        try:
            from .ackermann_gcs import AckermannGCS
        except ImportError:
            from ackermann_gcs import AckermannGCS

        return AckermannGCS.flat_output_to_state(
            flat_output,
            flat_derivs,
            self.wheelbase
        )

    def get_control(self, t: float) -> np.ndarray:
        """
        获取t时刻的控制输入

        Args:
            t: 实际时间

        Returns:
            控制 [a, ω]
        """
        flat_output = self.value(t)
        flat_derivs = [
            self.EvalDerivative(t, 1),
            self.EvalDerivative(t, 2)
        ]

        # 运行时导入，避免循环依赖
        try:
            from .ackermann_gcs import AckermannGCS
        except ImportError:
            from ackermann_gcs import AckermannGCS

        return AckermannGCS.flat_output_to_control(
            flat_output,
            flat_derivs,
            self.wheelbase
        )

    def start_time(self) -> float:
        """获取轨迹起始时间"""
        return self.time_trajectory.value(self.start_s)[0, 0]

    def end_time(self) -> float:
        """获取轨迹结束时间"""
        return self.time_trajectory.value(self.end_s)[0, 0]

    def validateSteeringConstraints(
        self,
        delta_min: float,
        delta_max: float,
        num_samples: int = 1000
    ) -> dict:
        """
        验证轨迹的转向角约束

        Args:
            delta_min: 最小转向角（弧度）
            delta_max: 最大转向角（弧度）
            num_samples: 采样点数

        Returns:
            验证结果字典，包含：
            - delta_min: 最小转向角（弧度）
            - delta_max: 最大转向角（弧度）
            - delta_mean: 平均转向角（弧度）
            - violations: 违反约束次数
            - violation_rate: 违反率
            - max_violation: 最大违反量（弧度）
            - violation_details: 违反详情列表（最多10条）
        """
        t_start = self.start_time()
        t_end = self.end_time()
        t_samples = np.linspace(t_start, t_end, num_samples)

        steering_angles = []
        violations_info = []

        for t in t_samples:
            state = self.get_state(t)
            delta = state[4]  # δ = state[4]
            steering_angles.append(delta)

            # 检查是否违反约束
            if delta < delta_min:
                violations_info.append({
                    't': t,
                    'delta': delta,
                    'bound': delta_min,
                    'violation': delta_min - delta,
                    'type': 'lower'
                })
            elif delta > delta_max:
                violations_info.append({
                    't': t,
                    'delta': delta,
                    'bound': delta_max,
                    'violation': delta - delta_max,
                    'type': 'upper'
                })

        # 计算统计信息
        max_violation = 0
        if violations_info:
            max_violation = max([v['violation'] for v in violations_info])

        return {
            'delta_min': np.min(steering_angles),
            'delta_max': np.max(steering_angles),
            'delta_mean': np.mean(steering_angles),
            'violations': len(violations_info),
            'violation_rate': len(violations_info) / num_samples,
            'max_violation': max_violation,
            'violation_details': violations_info[:10]  # 最多返回10条
        }

    def rows(self) -> int:
        """获取轨迹维度"""
        return self.flat_trajectory.rows()

    def cols(self) -> int:
        """获取轨迹列数"""
        return self.flat_trajectory.cols()
