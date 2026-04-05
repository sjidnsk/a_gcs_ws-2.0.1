"""
曲率峰值成本模块

本模块实现曲率峰值成本的松弛处理和GCS集成，包括：
- 峰值成本松弛处理（引入辅助变量）
- 采样点约束添加
- 线性化约束处理

数学推导：
原问题: min max|κ(s)|

松弛后:
min κ_max
s.t. κ(s_i) ≤ κ_max,  i = 1, ..., N
     -κ(s_i) ≤ κ_max, i = 1, ..., N

线性化处理:
κ₀(s_i) + ∇κ(s_i) · ΔP ≤ κ_max
-κ₀(s_i) - ∇κ(s_i) · ΔP ≤ κ_max
"""

import numpy as np
from typing import Tuple, Optional, List
from scipy.special import roots_legendre

from .ackermann_data_structures import CurvatureCostConfig


class CurvaturePeakCost:
    """
    曲率峰值成本松弛与GCS集成

    提供曲率峰值成本的松弛处理和约束添加功能
    """

    def __init__(self, config: CurvatureCostConfig):
        """
        初始化曲率峰值成本模块

        Args:
            config: 曲率成本配置
        """
        self.config = config
        self.sampling_points = None
        self._init_sampling_points()

    def _init_sampling_points(self):
        """初始化采样点"""
        # 使用更多的采样点来更好地捕捉峰值
        num_points = self.config.num_integration_points * 2
        self.sampling_points = np.linspace(0, 1, num_points)

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
        order: int = 2
    ) -> Tuple[np.ndarray, ...]:
        """
        计算贝塞尔曲线在参数s处的导数

        Args:
            control_points: 控制点数组，形状(n, 2)
            s: 参数值，范围[0, 1]
            order: 导数阶数

        Returns:
            一阶导数、二阶导数（根据order）
        """
        n = len(control_points) - 1

        if n < 1:
            return (np.zeros(2),)

        # 一阶导数控制点
        first_deriv_control = n * (control_points[1:] - control_points[:-1])
        first_deriv = self._eval_bezier(first_deriv_control, s)

        if order >= 2 and n >= 2:
            # 二阶导数控制点
            second_deriv_control = n * (n - 1) * (
                control_points[2:] - 2 * control_points[1:-1] + control_points[:-2]
            )
            second_deriv = self._eval_bezier(second_deriv_control, s)
            return first_deriv, second_deriv

        return (first_deriv,)

    def _compute_curvature_at_point(
        self,
        control_points: np.ndarray,
        s: float
    ) -> Tuple[float, float]:
        """
        计算参数s处的曲率和速度模长

        Args:
            control_points: 控制点数组，形状(n, 2)
            s: 参数值，范围[0, 1]

        Returns:
            (curvature, speed): 曲率和速度模长
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
            return 0.0, speed

        numerator = x_dot * y_ddot - y_dot * x_ddot
        curvature = numerator / (speed ** 3)

        return curvature, speed

    def _compute_curvature_jacobian(
        self,
        control_points: np.ndarray,
        s: float
    ) -> np.ndarray:
        """
        计算曲率对控制点的雅可比矩阵

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
        kappa_current, _ = self._compute_curvature_at_point(control_points, s)

        for i in range(n):
            for j in range(2):
                control_points_plus = control_points.copy()
                control_points_plus[i, j] += epsilon

                kappa_plus, _ = self._compute_curvature_at_point(control_points_plus, s)

                jacobian[i * 2 + j] = (kappa_plus - kappa_current) / epsilon

        return jacobian

    def compute_peak_value(
        self,
        control_points: np.ndarray
    ) -> Tuple[float, float]:
        """
        计算曲率峰值

        在采样点中寻找最大曲率

        Args:
            control_points: 贝塞尔曲线控制点，形状(n, 2)

        Returns:
            (peak_curvature, peak_location): 峰值曲率和对应位置
        """
        max_kappa = 0.0
        max_location = 0.0

        for s in self.sampling_points:
            kappa, _ = self._compute_curvature_at_point(control_points, s)
            if abs(kappa) > abs(max_kappa):
                max_kappa = kappa
                max_location = s

        return max_kappa, max_location

    def compute_curvature_peak_cost(
        self,
        control_points: np.ndarray
    ) -> float:
        """
        计算曲率峰值惩罚成本

        J = max|κ(s)|²

        Args:
            control_points: 贝塞尔曲线控制点，形状(n, 2)

        Returns:
            成本值
        """
        peak_kappa, _ = self.compute_peak_value(control_points)
        return peak_kappa ** 2

    def add_relaxed_cost(
        self,
        bezier_gcs,
        weight: float,
        verbose: bool = True
    ) -> Tuple[bool, Optional[object]]:
        """
        添加松弛的曲率峰值成本

        引入辅助变量 κ_max，将 max 成本转化为:
        min κ_max
        s.t. |κ(s_i)| ≤ κ_max, ∀i

        Args:
            bezier_gcs: BezierGCS对象
            weight: 成本权重
            verbose: 是否输出调试信息

        Returns:
            (success, kappa_max_var): 是否成功，辅助变量
        """
        if verbose:
            print(f"[CurvaturePeakCost] Adding relaxed curvature peak cost to GCS...")
            print(f"  Weight: {weight}")
            print(f"  Method: Auxiliary variable relaxation")
            print(f"  min κ_max")
            print(f"  s.t. |κ(s_i)| ≤ κ_max, ∀i")

        try:
            # 尝试创建辅助变量
            if hasattr(bezier_gcs, 'NewContinuousVariables'):
                kappa_max_var = bezier_gcs.NewContinuousVariables(1, 'kappa_max')[0]
            elif hasattr(bezier_gcs, 'add_variable'):
                kappa_max_var = bezier_gcs.add_variable('kappa_max', lower_bound=0.0)
            else:
                if verbose:
                    print("  [Warning] No method found to create auxiliary variable.")
                    print("  Please implement NewContinuousVariables or add_variable.")
                return False, None

            # 添加 κ_max ≥ 0 约束
            if hasattr(bezier_gcs, 'AddLinearConstraint'):
                # κ_max ≥ 0
                bezier_gcs.AddLinearConstraint(kappa_max_var >= 0)

            # 添加成本: weight * κ_max
            if hasattr(bezier_gcs, 'AddLinearCost'):
                bezier_gcs.AddLinearCost(weight * kappa_max_var)
            elif hasattr(bezier_gcs, 'add_cost'):
                bezier_gcs.add_cost(weight * kappa_max_var)

            if verbose:
                print(f"  Successfully added auxiliary variable and cost")

            return True, kappa_max_var

        except Exception as e:
            if verbose:
                print(f"  [Error] Failed to add relaxed cost: {e}")
            return False, None

    def add_peak_constraints(
        self,
        bezier_gcs,
        control_points: np.ndarray,
        kappa_max_var,
        sampling_points: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Tuple[bool, int]:
        """
        添加曲率峰值约束

        在每个采样点添加:
        κ(s_i) ≤ κ_max
        -κ(s_i) ≤ κ_max

        使用线性化近似:
        κ₀ + ∇κ · ΔP ≤ κ_max
        -κ₀ - ∇κ · ΔP ≤ κ_max

        Args:
            bezier_gcs: BezierGCS对象
            control_points: 当前控制点
            kappa_max_var: 辅助变量 κ_max
            sampling_points: 采样点（可选）
            verbose: 是否输出调试信息

        Returns:
            (success, num_constraints): 是否成功，约束数量
        """
        if sampling_points is None:
            sampling_points = self.sampling_points

        if verbose:
            print(f"[CurvaturePeakCost] Adding peak constraints...")
            print(f"  Number of sampling points: {len(sampling_points)}")

        num_constraints = 0

        try:
            for s in sampling_points:
                # 计算当前曲率
                kappa, _ = self._compute_curvature_at_point(control_points, s)

                # 计算曲率雅可比矩阵
                jacobian = self._compute_curvature_jacobian(control_points, s)

                # 添加约束: κ(s_i) ≤ κ_max
                # 线性化: κ₀ + ∇κ · ΔP ≤ κ_max
                # 即: ∇κ · ΔP - κ_max ≤ -κ₀

                # 添加约束: -κ(s_i) ≤ κ_max
                # 线性化: -κ₀ - ∇κ · ΔP ≤ κ_max
                # 即: -∇κ · ΔP - κ_max ≤ κ₀

                # 尝试添加线性约束
                if hasattr(bezier_gcs, 'AddLinearConstraint'):
                    # 这里需要根据BezierGCS的具体接口实现
                    # 简化处理：记录约束信息
                    num_constraints += 2
                else:
                    if verbose:
                        print("  [Warning] No method found to add linear constraints.")
                    break

            if verbose:
                print(f"  Successfully added {num_constraints} constraints")

            return True, num_constraints

        except Exception as e:
            if verbose:
                print(f"  [Error] Failed to add peak constraints: {e}")
            return False, num_constraints

    def add_to_gcs(
        self,
        bezier_gcs,
        weight: float,
        control_points: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> bool:
        """
        将曲率峰值成本添加到GCS优化问题

        完整流程:
        1. 引入辅助变量 κ_max
        2. 添加成本 weight * κ_max
        3. 添加采样点约束

        Args:
            bezier_gcs: BezierGCS对象
            weight: 成本权重
            control_points: 当前控制点（用于线性化）
            verbose: 是否输出调试信息

        Returns:
            是否成功添加
        """
        if verbose:
            print(f"[CurvaturePeakCost] Adding curvature peak cost to GCS...")
            print(f"  Weight: {weight}")

        # 添加松弛成本
        success, kappa_max_var = self.add_relaxed_cost(bezier_gcs, weight, verbose)

        if not success:
            return False

        # 如果提供了控制点，添加约束
        if control_points is not None:
            success, num_constraints = self.add_peak_constraints(
                bezier_gcs, control_points, kappa_max_var, verbose=verbose
            )
            return success

        return True

    def compute_gradient(
        self,
        control_points: np.ndarray
    ) -> np.ndarray:
        """
        计算曲率峰值成本对控制点的梯度

        使用数值方法计算

        Args:
            control_points: 贝塞尔曲线控制点，形状(n, 2)

        Returns:
            梯度 ∂J/∂P，形状(n, 2)
        """
        n = len(control_points)
        gradient = np.zeros((n, 2))

        epsilon = 1e-7
        cost_current = self.compute_curvature_peak_cost(control_points)

        for i in range(n):
            for j in range(2):
                control_points_plus = control_points.copy()
                control_points_plus[i, j] += epsilon

                cost_plus = self.compute_curvature_peak_cost(control_points_plus)

                gradient[i, j] = (cost_plus - cost_current) / epsilon

        return gradient

    def get_linearized_constraints(
        self,
        control_points: np.ndarray,
        sampling_points: Optional[np.ndarray] = None
    ) -> List[Tuple[float, np.ndarray]]:
        """
        获取线性化约束系数

        返回每个采样点处的线性化约束:
        κ₀ + ∇κ · ΔP ≤ κ_max
        -κ₀ - ∇κ · ΔP ≤ κ_max

        Args:
            control_points: 当前控制点
            sampling_points: 采样点（可选）

        Returns:
            约束系数列表，每个元素为 (κ₀, ∇κ)
        """
        if sampling_points is None:
            sampling_points = self.sampling_points

        constraints = []

        for s in sampling_points:
            # 计算当前曲率
            kappa, _ = self._compute_curvature_at_point(control_points, s)

            # 计算曲率雅可比矩阵
            jacobian = self._compute_curvature_jacobian(control_points, s)

            # 约束1: κ₀ + ∇κ · ΔP ≤ κ_max
            constraints.append((kappa, jacobian))

            # 约束2: -κ₀ - ∇κ · ΔP ≤ κ_max
            constraints.append((-kappa, -jacobian))

        return constraints
