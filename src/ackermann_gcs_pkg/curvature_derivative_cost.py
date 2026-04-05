"""
曲率导数成本模块

本模块实现曲率导数成本的解析计算和GCS集成，包括：
- 三阶导数计算
- 曲率导数解析公式
- GCS成本集成（使用代理成本）

数学推导：
曲率导数: dκ/ds = (dκ/dt) / ||r'(s)||

其中:
dκ/dt = (n'd - nd') / d²

n = ẋÿ - ẏẍ
n' = ẋÿ' + ẋ'ÿ - ẏẍ' - ẏ'ẍ

d = (ẋ² + ẏ²)^(3/2)
d' = 1.5 * (ẋ² + ẏ²)^0.5 * (2ẋẋ' + 2ẏÿ')

代理成本（凸近似）:
J_proxy = ∫||r'''(s)||² ds

这是凸的，因为r'''(s)是控制点的线性函数
"""

import numpy as np
from typing import Tuple, Optional, List
from scipy.special import roots_legendre

from .ackermann_data_structures import (
    CurvatureCostConfig,
    CurvatureDerivatives,
)


class CurvatureDerivativeCost:
    """
    曲率导数成本计算与GCS集成

    提供曲率导数的解析计算和GCS成本添加功能
    """

    def __init__(self, config: CurvatureCostConfig):
        """
        初始化曲率导数成本模块

        Args:
            config: 曲率成本配置
        """
        self.config = config
        self.sampling_points = None
        self.integration_weights = None
        self._init_integration_points()

    def _init_integration_points(self):
        """初始化数值积分采样点和权重"""
        if self.config.integration_method == "gauss_legendre":
            points, weights = roots_legendre(self.config.num_integration_points)
            self.sampling_points = (points + 1) / 2
            self.integration_weights = weights / 2
        elif self.config.integration_method == "trapezoid":
            self.sampling_points = np.linspace(0, 1, self.config.num_integration_points)
            self.integration_weights = np.ones(self.config.num_integration_points) / (self.config.num_integration_points - 1)
            self.integration_weights[0] /= 2
            self.integration_weights[-1] /= 2
        elif self.config.integration_method == "simpson":
            n = self.config.num_integration_points
            if n % 2 == 0:
                n += 1
            self.sampling_points = np.linspace(0, 1, n)
            h = 1.0 / (n - 1)
            self.integration_weights = np.ones(n) * h / 3
            self.integration_weights[1:-1:2] = 4 * h / 3
            self.integration_weights[2:-1:2] = 2 * h / 3
        else:
            raise ValueError(f"Unknown integration method: {self.config.integration_method}")

    def _eval_bezier(self, control_points: np.ndarray, s: float) -> np.ndarray:
        """
        使用de Casteljau算法计算贝塞尔曲线值

        Args:
            control_points: 控制点数组，形状(n, 2)
            s: 参数值，范围[0, 1]

        Returns:
            曲线点，形状(2,)
        """
        points = control_points.copy()
        n = len(points)

        for k in range(1, n):
            for i in range(n - k):
                points[i] = (1 - s) * points[i] + s * points[i + 1]

        return points[0]

    def compute_third_derivative(
        self,
        control_points: np.ndarray,
        s: float
    ) -> np.ndarray:
        """
        计算三阶导数 r'''(s)

        三阶导数控制点: n(n-1)(n-2) * Δ³P

        其中 Δ³P 是三阶前向差分

        Args:
            control_points: 控制点数组，形状(n, 2)
            s: 参数值，范围[0, 1]

        Returns:
            三阶导数，形状(2,)
        """
        n = len(control_points) - 1

        if n < 3:
            return np.zeros(2)

        # 三阶导数控制点
        # Δ³P_i = P_{i+3} - 3*P_{i+2} + 3*P_{i+1} - P_i
        third_deriv_control = n * (n - 1) * (n - 2) * (
            control_points[3:] - 3 * control_points[2:-1] +
            3 * control_points[1:-2] - control_points[:-3]
        )

        return self._eval_bezier(third_deriv_control, s)

    def compute_bezier_derivatives(
        self,
        control_points: np.ndarray,
        s: float,
        order: int = 3
    ) -> Tuple[np.ndarray, ...]:
        """
        计算贝塞尔曲线在参数s处的各阶导数

        Args:
            control_points: 控制点数组，形状(n, 2)
            s: 参数值，范围[0, 1]
            order: 最高导数阶数（1, 2, 或 3）

        Returns:
            一阶导数、二阶导数、三阶导数（根据order）
        """
        n = len(control_points) - 1

        if n < 1:
            zero = np.zeros(2)
            if order == 1:
                return (zero,)
            elif order == 2:
                return (zero, zero)
            else:
                return (zero, zero, zero)

        # 一阶导数控制点
        first_deriv_control = n * (control_points[1:] - control_points[:-1])
        first_deriv = self._eval_bezier(first_deriv_control, s)

        if order >= 2 and n >= 2:
            # 二阶导数控制点
            second_deriv_control = n * (n - 1) * (
                control_points[2:] - 2 * control_points[1:-1] + control_points[:-2]
            )
            second_deriv = self._eval_bezier(second_deriv_control, s)

            if order >= 3 and n >= 3:
                # 三阶导数控制点
                third_deriv_control = n * (n - 1) * (n - 2) * (
                    control_points[3:] - 3 * control_points[2:-1] +
                    3 * control_points[1:-2] - control_points[:-3]
                )
                third_deriv = self._eval_bezier(third_deriv_control, s)
                return first_deriv, second_deriv, third_deriv

            return first_deriv, second_deriv

        return (first_deriv,)

    def compute_curvature_derivative(
        self,
        control_points: np.ndarray,
        s: float
    ) -> float:
        """
        计算参数s处的曲率导数 dκ/ds

        解析公式:
        dκ/ds = (dκ/dt) / ||r'(s)||

        其中 dκ/dt 使用三阶导数计算:
        dκ/dt = (n'd - nd') / d²

        n = ẋÿ - ẏẍ
        n' = ẋÿ' + ẋ'ÿ - ẏẍ' - ẏ'ẍ

        d = (ẋ² + ẏ²)^(3/2)
        d' = 1.5 * (ẋ² + ẏ²)^0.5 * (2ẋẋ' + 2ẏÿ')

        Args:
            control_points: 控制点数组，形状(n, 2)
            s: 参数值，范围[0, 1]

        Returns:
            曲率导数 dκ/ds
        """
        # 计算一阶、二阶、三阶导数
        first_deriv, second_deriv, third_deriv = self.compute_bezier_derivatives(
            control_points, s, order=3
        )

        x_dot = first_deriv[0]
        y_dot = first_deriv[1]
        x_ddot = second_deriv[0]
        y_ddot = second_deriv[1]
        x_dddot = third_deriv[0]
        y_dddot = third_deriv[1]

        # 计算速度
        speed = np.sqrt(x_dot**2 + y_dot**2)

        if speed < self.config.numerical_tolerance:
            return 0.0

        # 计算曲率公式的分子和分母
        n = x_dot * y_ddot - y_dot * x_ddot
        d = speed ** 3

        # 计算n'和d'
        # n' = ẋÿ' + ẋ'ÿ - ẏẍ' - ẏ'ẍ
        # 注意：ẋ' = ẍ, ÿ' = ÿ, ẍ' = x''', ẏ' = y'''
        n_prime = x_dot * y_dddot + x_ddot * y_ddot - y_dot * x_dddot - y_ddot * x_ddot

        # d' = 1.5 * (ẋ² + ẏ²)^0.5 * (2ẋẋ' + 2ẏÿ')
        d_prime = 1.5 * speed * (2 * x_dot * x_ddot + 2 * y_dot * y_ddot)

        # dκ/dt = (n'd - nd') / d²
        dkappa_dt = (n_prime * d - n * d_prime) / (d ** 2)

        # dκ/ds = (dκ/dt) / ||r'(s)||
        dkappa_ds = dkappa_dt / speed

        return dkappa_ds

    def compute_curvature_and_derivative(
        self,
        control_points: np.ndarray,
        s: float
    ) -> CurvatureDerivatives:
        """
        计算曲率及曲率导数

        Args:
            control_points: 控制点数组，形状(n, 2)
            s: 参数值，范围[0, 1]

        Returns:
            CurvatureDerivatives: 曲率及各阶导数
        """
        # 计算各阶导数
        first_deriv, second_deriv, third_deriv = self.compute_bezier_derivatives(
            control_points, s, order=3
        )

        x_dot = first_deriv[0]
        y_dot = first_deriv[1]
        x_ddot = second_deriv[0]
        y_ddot = second_deriv[1]
        x_dddot = third_deriv[0]
        y_dddot = third_deriv[1]

        # 计算速度
        speed = np.sqrt(x_dot**2 + y_dot**2)

        if speed < self.config.numerical_tolerance:
            return CurvatureDerivatives(
                curvature=0.0,
                curvature_derivative=0.0,
                first_deriv=first_deriv,
                second_deriv=second_deriv,
                third_deriv=third_deriv,
                speed=speed
            )

        # 计算曲率
        numerator = x_dot * y_ddot - y_dot * x_ddot
        denominator = speed ** 3
        kappa = numerator / denominator

        # 计算曲率导数
        n = numerator
        d = denominator
        n_prime = x_dot * y_dddot + x_ddot * y_ddot - y_dot * x_dddot - y_ddot * x_ddot
        d_prime = 1.5 * speed * (2 * x_dot * x_ddot + 2 * y_dot * y_ddot)

        dkappa_dt = (n_prime * d - n * d_prime) / (d ** 2)
        dkappa_ds = dkappa_dt / speed

        return CurvatureDerivatives(
            curvature=kappa,
            curvature_derivative=dkappa_ds,
            first_deriv=first_deriv,
            second_deriv=second_deriv,
            third_deriv=third_deriv,
            speed=speed
        )

    def compute_curvature_derivative_cost(
        self,
        control_points: np.ndarray
    ) -> float:
        """
        计算曲率导数平方积分成本

        J = ∫(dκ/ds)² * ||r'(s)|| ds

        使用解析曲率导数计算

        Args:
            control_points: 贝塞尔曲线控制点，形状(n, 2)

        Returns:
            成本值
        """
        cost = 0.0

        for s_i, w_i in zip(self.sampling_points, self.integration_weights):
            # 计算曲率导数
            dkappa_ds = self.compute_curvature_derivative(control_points, s_i)

            # 计算速度（用于弧长参数化）
            first_deriv = self.compute_bezier_derivatives(control_points, s_i, order=1)[0]
            speed = np.linalg.norm(first_deriv)

            # 积分：∫(dκ/ds)² * ||r'(s)|| ds
            cost += w_i * dkappa_ds**2 * speed

        return cost

    def add_to_gcs(
        self,
        bezier_gcs,
        weight: float,
        verbose: bool = True
    ) -> bool:
        """
        将曲率导数成本添加到GCS优化问题

        使用代理成本: J = ∫||r'''(s)||² ds

        这是凸的，因为r'''(s)是控制点的线性函数

        Args:
            bezier_gcs: BezierGCS对象
            weight: 成本权重
            verbose: 是否输出调试信息

        Returns:
            是否成功添加
        """
        if verbose:
            print(f"[CurvatureDerivativeCost] Adding curvature derivative cost to GCS...")
            print(f"  Weight: {weight}")
            print(f"  Method: Proxy cost using third derivative squared")
            print(f"  J_proxy = ∫||r'''(s)||² ds")

        try:
            # 尝试使用BezierGCS的方法添加三阶导数成本
            if hasattr(bezier_gcs, 'addDerivativeCost'):
                # 添加三阶导数成本
                bezier_gcs.addDerivativeCost(order=3, weight=weight)
                if verbose:
                    print("  Successfully added via addDerivativeCost(order=3)")
                return True
            elif hasattr(bezier_gcs, 'add_derivative_regularization'):
                # 添加导数正则化
                bezier_gcs.add_derivative_regularization(order=3, weight=weight)
                if verbose:
                    print("  Successfully added via add_derivative_regularization(order=3)")
                return True
            elif hasattr(bezier_gcs, 'addPathSmoothnessCost'):
                # 添加路径平滑度成本
                bezier_gcs.addPathSmoothnessCost(order=3, weight=weight)
                if verbose:
                    print("  Successfully added via addPathSmoothnessCost(order=3)")
                return True
            else:
                if verbose:
                    print("  [Warning] No suitable method found in BezierGCS for curvature derivative cost.")
                    print("  Available methods: addDerivativeCost, add_derivative_regularization, addPathSmoothnessCost")
                    print("  Please implement one of these methods.")
                return False
        except Exception as e:
            if verbose:
                print(f"  [Error] Failed to add curvature derivative cost: {e}")
            return False

    def compute_gradient(
        self,
        control_points: np.ndarray
    ) -> np.ndarray:
        """
        计算曲率导数成本对控制点的梯度

        ∂J/∂P = ∫2(dκ/ds) * ∂(dκ/ds)/∂P * ||r'(s)|| ds

        使用数值方法计算

        Args:
            control_points: 贝塞尔曲线控制点，形状(n, 2)

        Returns:
            梯度 ∂J/∂P，形状(n, 2)
        """
        n = len(control_points)
        gradient = np.zeros((n, 2))

        epsilon = 1e-7

        for i in range(n):
            for j in range(2):
                # 前向差分
                control_points_plus = control_points.copy()
                control_points_plus[i, j] += epsilon

                cost_plus = self.compute_curvature_derivative_cost(control_points_plus)
                cost_current = self.compute_curvature_derivative_cost(control_points)

                gradient[i, j] = (cost_plus - cost_current) / epsilon

        return gradient

    def compute_jacobian(
        self,
        control_points: np.ndarray,
        s: float
    ) -> np.ndarray:
        """
        计算曲率导数对控制点的雅可比矩阵

        ∂(dκ/ds)/∂P

        使用数值方法计算

        Args:
            control_points: 控制点数组，形状(n, 2)
            s: 参数值，范围[0, 1]

        Returns:
            雅可比矩阵，形状(n*2,)
        """
        n = len(control_points)
        jacobian = np.zeros(n * 2)

        epsilon = 1e-7
        dkappa_ds_current = self.compute_curvature_derivative(control_points, s)

        for i in range(n):
            for j in range(2):
                control_points_plus = control_points.copy()
                control_points_plus[i, j] += epsilon

                dkappa_ds_plus = self.compute_curvature_derivative(control_points_plus, s)

                jacobian[i * 2 + j] = (dkappa_ds_plus - dkappa_ds_current) / epsilon

        return jacobian
