"""
解析梯度计算器

本模块实现成本对控制点的解析梯度计算，替代数值差分方法，包括：
- 曲率平方成本梯度
- 曲率导数成本梯度
- 曲率峰值成本梯度
- 梯度缓存机制

数学推导：
总成本: J = w₁ * J₁ + w₂ * J₂ + w₃ * J₃

其中:
J₁ = ∫κ²(s)ds (曲率平方)
J₂ = ∫(dκ/ds)²ds (曲率导数平方)
J₃ = max|κ(s)| (曲率峰值)

梯度:
∂J/∂P = w₁ * ∂J₁/∂P + w₂ * ∂J₂/∂P + w₃ * ∂J₃/∂P

∂J₁/∂P = ∫2κ(s) * ∂κ/∂P ds
∂J₂/∂P = ∫2(dκ/ds) * ∂(dκ/ds)/∂P ds
∂J₃/∂P = ∂(max|κ|)/∂P (在峰值点处)
"""

import numpy as np
from typing import Dict, Optional, Tuple
import hashlib
from scipy.special import roots_legendre

from .ackermann_data_structures import (
    CurvatureCostConfig,
    CurvatureCostWeights,
)


class AnalyticGradientCalculator:
    """
    解析梯度计算器

    提供成本对控制点的解析梯度计算，支持缓存机制
    """

    def __init__(
        self,
        config: CurvatureCostConfig,
        enable_cache: bool = True
    ):
        """
        初始化解析梯度计算器

        Args:
            config: 曲率成本配置
            enable_cache: 是否启用缓存
        """
        self.config = config
        self.enable_cache = enable_cache
        self.cache: Dict[str, np.ndarray] = {}
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

    def _get_cache_key(self, control_points: np.ndarray, prefix: str = "") -> str:
        """
        基于控制点生成缓存键

        Args:
            control_points: 控制点数组
            prefix: 缓存键前缀

        Returns:
            缓存键字符串
        """
        # 使用控制点的哈希值作为键
        point_hash = hashlib.md5(control_points.tobytes()).hexdigest()
        return f"{prefix}_{point_hash}"

    def clear_cache(self) -> None:
        """清除缓存"""
        self.cache.clear()

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

    def _compute_bezier_derivatives(
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
            order: 最高导数阶数

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

    def _compute_curvature_and_speed(
        self,
        control_points: np.ndarray,
        s: float
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        计算曲率、速度和导数

        Args:
            control_points: 控制点数组
            s: 参数值

        Returns:
            (curvature, speed, first_deriv, second_deriv)
        """
        first_deriv, second_deriv = self._compute_bezier_derivatives(
            control_points, s, order=2
        )

        x_dot = first_deriv[0]
        y_dot = first_deriv[1]
        x_ddot = second_deriv[0]
        y_ddot = second_deriv[1]

        speed = np.sqrt(x_dot**2 + y_dot**2)

        if speed < self.config.numerical_tolerance:
            return 0.0, speed, first_deriv, second_deriv

        numerator = x_dot * y_ddot - y_dot * x_ddot
        curvature = numerator / (speed ** 3)

        return curvature, speed, first_deriv, second_deriv

    def _compute_curvature_jacobian(
        self,
        control_points: np.ndarray,
        s: float
    ) -> np.ndarray:
        """
        计算曲率对控制点的雅可比矩阵

        使用数值方法计算

        Args:
            control_points: 控制点数组
            s: 参数值

        Returns:
            雅可比矩阵，形状(n*2,)
        """
        n = len(control_points)
        jacobian = np.zeros(n * 2)

        epsilon = 1e-7
        kappa_current, _, _, _ = self._compute_curvature_and_speed(control_points, s)

        for i in range(n):
            for j in range(2):
                control_points_plus = control_points.copy()
                control_points_plus[i, j] += epsilon

                kappa_plus, _, _, _ = self._compute_curvature_and_speed(control_points_plus, s)

                jacobian[i * 2 + j] = (kappa_plus - kappa_current) / epsilon

        return jacobian

    def compute_curvature_gradient(
        self,
        control_points: np.ndarray,
        sampling_points: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算曲率平方成本梯度

        ∂J₁/∂P = ∫2κ(s) * ∂κ/∂P * ||r'(s)|| ds

        使用数值积分: Σ w_i * 2κ(s_i) * ∂κ/∂P|_{s_i} * ||r'(s_i)||

        Args:
            control_points: 控制点数组，形状(n, 2)
            sampling_points: 积分采样点（可选）
            weights: 积分权重（可选）

        Returns:
            梯度向量，形状(n*2,)
        """
        # 检查缓存
        if self.enable_cache:
            cache_key = self._get_cache_key(control_points, "curvature_grad")
            if cache_key in self.cache:
                return self.cache[cache_key].copy()

        if sampling_points is None:
            sampling_points = self.sampling_points
            weights = self.integration_weights

        n = len(control_points)
        gradient = np.zeros(n * 2)

        # 计算积分
        for s_i, w_i in zip(sampling_points, weights):
            # 计算曲率
            kappa, speed, _, _ = self._compute_curvature_and_speed(control_points, s_i)

            # 计算雅可比矩阵
            jacobian = self._compute_curvature_jacobian(control_points, s_i)

            # 梯度贡献: 2κ(s) * ∂κ/∂P * ||r'(s)||
            gradient += w_i * 2 * kappa * jacobian * speed

        # 缓存结果
        if self.enable_cache:
            self.cache[cache_key] = gradient.copy()

        return gradient

    def compute_curvature_derivative_gradient(
        self,
        control_points: np.ndarray,
        sampling_points: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算曲率导数平方成本梯度

        ∂J₂/∂P = ∫2(dκ/ds) * ∂(dκ/ds)/∂P * ||r'(s)|| ds

        Args:
            control_points: 控制点数组，形状(n, 2)
            sampling_points: 积分采样点（可选）
            weights: 积分权重（可选）

        Returns:
            梯度向量，形状(n*2,)
        """
        # 检查缓存
        if self.enable_cache:
            cache_key = self._get_cache_key(control_points, "curvature_deriv_grad")
            if cache_key in self.cache:
                return self.cache[cache_key].copy()

        if sampling_points is None:
            sampling_points = self.sampling_points
            weights = self.integration_weights

        n = len(control_points)
        gradient = np.zeros(n * 2)

        epsilon = 1e-7

        # 计算积分
        for s_i, w_i in zip(sampling_points, weights):
            # 计算曲率导数
            dkappa_ds_current = self._compute_curvature_derivative(control_points, s_i)

            # 计算曲率导数对控制点的雅可比矩阵
            jacobian = np.zeros(n * 2)
            for idx in range(n):
                for j in range(2):
                    control_points_plus = control_points.copy()
                    control_points_plus[idx, j] += epsilon

                    dkappa_ds_plus = self._compute_curvature_derivative(control_points_plus, s_i)

                    jacobian[idx * 2 + j] = (dkappa_ds_plus - dkappa_ds_current) / epsilon

            # 计算速度
            first_deriv = self._compute_bezier_derivatives(control_points, s_i, order=1)[0]
            speed = np.linalg.norm(first_deriv)

            # 梯度贡献: 2(dκ/ds) * ∂(dκ/ds)/∂P * ||r'(s)||
            gradient += w_i * 2 * dkappa_ds_current * jacobian * speed

        # 缓存结果
        if self.enable_cache:
            self.cache[cache_key] = gradient.copy()

        return gradient

    def _compute_curvature_derivative(
        self,
        control_points: np.ndarray,
        s: float
    ) -> float:
        """
        计算曲率导数 dκ/ds

        Args:
            control_points: 控制点数组
            s: 参数值

        Returns:
            曲率导数
        """
        # 计算一阶、二阶、三阶导数
        first_deriv, second_deriv, third_deriv = self._compute_bezier_derivatives(
            control_points, s, order=3
        )

        x_dot = first_deriv[0]
        y_dot = first_deriv[1]
        x_ddot = second_deriv[0]
        y_ddot = second_deriv[1]
        x_dddot = third_deriv[0]
        y_dddot = third_deriv[1]

        speed = np.sqrt(x_dot**2 + y_dot**2)

        if speed < self.config.numerical_tolerance:
            return 0.0

        # 计算曲率公式的分子和分母
        n = x_dot * y_ddot - y_dot * x_ddot
        d = speed ** 3

        # 计算n'和d'
        n_prime = x_dot * y_dddot + x_ddot * y_ddot - y_dot * x_dddot - y_ddot * x_ddot
        d_prime = 1.5 * speed * (2 * x_dot * x_ddot + 2 * y_dot * y_ddot)

        # dκ/dt = (n'd - nd') / d²
        dkappa_dt = (n_prime * d - n * d_prime) / (d ** 2)

        # dκ/ds = (dκ/dt) / ||r'(s)||
        dkappa_ds = dkappa_dt / speed

        return dkappa_ds

    def compute_curvature_peak_gradient(
        self,
        control_points: np.ndarray,
        sampling_points: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算曲率峰值成本梯度

        ∂J₃/∂P = ∂(max|κ|)/∂P

        在峰值点处计算梯度

        Args:
            control_points: 控制点数组，形状(n, 2)
            sampling_points: 采样点（可选）

        Returns:
            梯度向量，形状(n*2,)
        """
        # 检查缓存
        if self.enable_cache:
            cache_key = self._get_cache_key(control_points, "curvature_peak_grad")
            if cache_key in self.cache:
                return self.cache[cache_key].copy()

        if sampling_points is None:
            sampling_points = np.linspace(0, 1, self.config.num_integration_points * 2)

        n = len(control_points)

        # 找到峰值点
        max_kappa = 0.0
        peak_location = 0.0

        for s in sampling_points:
            kappa, _, _, _ = self._compute_curvature_and_speed(control_points, s)
            if abs(kappa) > abs(max_kappa):
                max_kappa = kappa
                peak_location = s

        # 在峰值点处计算梯度
        # ∂(max|κ|)/∂P = sign(κ_peak) * ∂κ/∂P|_{s_peak}
        jacobian = self._compute_curvature_jacobian(control_points, peak_location)
        gradient = np.sign(max_kappa) * jacobian

        # 缓存结果
        if self.enable_cache:
            self.cache[cache_key] = gradient.copy()

        return gradient

    def compute_gradient(
        self,
        control_points: np.ndarray,
        weights: CurvatureCostWeights
    ) -> np.ndarray:
        """
        计算总成本对控制点的梯度

        ∂J/∂P = w₁ * ∂J₁/∂P + w₂ * ∂J₂/∂P + w₃ * ∂J₃/∂P

        Args:
            control_points: 控制点数组，形状(n, 2)
            weights: 成本权重配置

        Returns:
            梯度向量，形状(n*2,)
        """
        # 检查缓存
        if self.enable_cache:
            cache_key = self._get_cache_key(control_points, "total_grad")
            if cache_key in self.cache:
                return self.cache[cache_key].copy()

        n = len(control_points)
        gradient = np.zeros(n * 2)

        # 曲率平方成本梯度
        if weights.curvature_squared > 0:
            grad_curvature = self.compute_curvature_gradient(control_points)
            gradient += weights.curvature_squared * grad_curvature

        # 曲率导数成本梯度
        if weights.curvature_derivative > 0:
            grad_derivative = self.compute_curvature_derivative_gradient(control_points)
            gradient += weights.curvature_derivative * grad_derivative

        # 曲率峰值成本梯度
        if weights.curvature_peak > 0:
            grad_peak = self.compute_curvature_peak_gradient(control_points)
            gradient += weights.curvature_peak * grad_peak

        # 缓存结果
        if self.enable_cache:
            self.cache[cache_key] = gradient.copy()

        return gradient

    def compute_gradient_2d(
        self,
        control_points: np.ndarray,
        weights: CurvatureCostWeights
    ) -> np.ndarray:
        """
        计算总成本对控制点的梯度（2D形式）

        Args:
            control_points: 控制点数组，形状(n, 2)
            weights: 成本权重配置

        Returns:
            梯度矩阵，形状(n, 2)
        """
        gradient_flat = self.compute_gradient(control_points, weights)
        n = len(control_points)
        return gradient_flat.reshape(n, 2)

    def compare_with_numerical_gradient(
        self,
        control_points: np.ndarray,
        weights: CurvatureCostWeights,
        epsilon: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        对比解析梯度与数值梯度

        Args:
            control_points: 控制点数组
            weights: 成本权重配置
            epsilon: 数值差分步长

        Returns:
            (analytic_grad, numerical_grad, max_error)
        """
        n = len(control_points)

        # 解析梯度
        analytic_grad = self.compute_gradient(control_points, weights)

        # 数值梯度
        numerical_grad = np.zeros(n * 2)

        # 计算当前成本
        from .curvature_cost_module import CurvatureCostModule
        cost_module = CurvatureCostModule(self.config)

        def compute_total_cost(cp):
            cost = 0.0
            if weights.curvature_squared > 0:
                cost += weights.curvature_squared * cost_module.compute_curvature_squared_cost(cp)
            if weights.curvature_derivative > 0:
                cost += weights.curvature_derivative * cost_module.compute_curvature_derivative_cost(cp)
            if weights.curvature_peak > 0:
                cost += weights.curvature_peak * cost_module.compute_curvature_peak_cost(cp)
            return cost

        cost_current = compute_total_cost(control_points)

        for i in range(n):
            for j in range(2):
                control_points_plus = control_points.copy()
                control_points_plus[i, j] += epsilon

                cost_plus = compute_total_cost(control_points_plus)

                numerical_grad[i * 2 + j] = (cost_plus - cost_current) / epsilon

        # 计算最大误差
        max_error = np.max(np.abs(analytic_grad - numerical_grad))

        return analytic_grad, numerical_grad, max_error
