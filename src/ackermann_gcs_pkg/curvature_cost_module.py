"""
曲率惩罚成本模块

本模块实现曲率惩罚成本的计算和添加功能，包括：
- 曲率平方积分成本
- 曲率导数平方积分成本
- 曲率峰值惩罚成本
- 凸松弛方法
- 高精度数值积分
- SCP线性化支持
- 解析梯度计算
"""

import numpy as np
from typing import Tuple, Optional, List
from scipy.special import roots_legendre

from .ackermann_data_structures import (
    CurvatureCostConfig,
    CurvatureCostWeights,
    LinearizedCostCoefficients,
)

# 导入数值安全工具
from .numerical_safety_utils import safe_curvature_calculation, DEFAULT_SMALL_VALUE

# 导入接口
from .cost_calculator_interface import CostCalculatorInterface

# 导入新模块
from .curvature_cost_linearizer import CurvatureCostLinearizer
from .curvature_derivative_cost import CurvatureDerivativeCost
from .curvature_peak_cost import CurvaturePeakCost
from .curvature_squared_cost_calculator import CurvatureSquaredCostCalculator
from .analytic_gradient_calculator import AnalyticGradientCalculator


class CurvatureCostModule:
    """
    曲率成本模块

    提供曲率惩罚成本的计算和添加功能，用于引导生成更平滑的轨迹。
    """

    def __init__(
        self,
        config: CurvatureCostConfig,
        cost_calculators: Optional[List[CostCalculatorInterface]] = None
    ):
        """
        初始化曲率成本模块

        Args:
            config: 曲率成本配置
            cost_calculators: 成本计算器列表 (可选，用于依赖注入)
        """
        self.config = config
        self.sampling_points = None
        self.weights = None
        self._init_integration_points()

        # 初始化新模块
        self.linearizer = CurvatureCostLinearizer(config)
        self.derivative_cost = CurvatureDerivativeCost(config)
        self.peak_cost = CurvaturePeakCost(config)
        
        # 初始化成本计算器
        if cost_calculators is None:
            # 默认创建成本计算器
            self.cost_calculators = [
                CurvatureSquaredCostCalculator(config),
                CurvatureDerivativeCost(config),
                CurvaturePeakCost(config),
            ]
        else:
            self.cost_calculators = cost_calculators
        
        # 初始化梯度计算器，注入成本计算器
        self.gradient_calculator = AnalyticGradientCalculator(
            config,
            cost_calculators=self.cost_calculators
        )

    def _init_integration_points(self):
        """
        初始化数值积分采样点和权重

        根据配置的积分方法，初始化相应的采样点和权重。
        目前主要支持高斯-勒让德积分。
        """
        if self.config.integration_method == "gauss_legendre":
            # 使用高斯-勒让德积分
            # roots_legendre返回区间[-1, 1]上的采样点和权重
            points, weights = roots_legendre(self.config.num_integration_points)
            # 转换到区间[0, 1]
            self.sampling_points = (points + 1) / 2
            self.weights = weights / 2
        elif self.config.integration_method == "trapezoid":
            # 梯形积分
            self.sampling_points = np.linspace(0, 1, self.config.num_integration_points)
            self.weights = np.ones(self.config.num_integration_points) / (self.config.num_integration_points - 1)
            self.weights[0] /= 2
            self.weights[-1] /= 2
        elif self.config.integration_method == "simpson":
            # 辛普森积分（需要奇数个点）
            n = self.config.num_integration_points
            if n % 2 == 0:
                n += 1  # 确保奇数个点
            self.sampling_points = np.linspace(0, 1, n)
            h = 1.0 / (n - 1)
            self.weights = np.ones(n) * h / 3
            self.weights[1:-1:2] = 4 * h / 3  # 奇数索引
            self.weights[2:-1:2] = 2 * h / 3  # 偶数索引
        else:
            raise ValueError(f"Unknown integration method: {self.config.integration_method}")

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
            order: 导数阶数（1=一阶导数，2=二阶导数）

        Returns:
            一阶导数、二阶导数（如果order=2）
        """
        n = len(control_points) - 1  # 贝塞尔曲线阶数

        # 一阶导数：n * (P_{i+1} - P_i) * B_{n-1,i}(s)
        if n < 1:
            return (np.zeros(2),)

        # 计算一阶导数的控制点
        first_deriv_control = n * (control_points[1:] - control_points[:-1])

        # 计算一阶导数值
        first_deriv = self._eval_bezier(first_deriv_control, s)

        if order >= 2 and n >= 2:
            # 二阶导数：n * (n-1) * (P_{i+2} - 2*P_{i+1} + P_i) * B_{n-2,i}(s)
            second_deriv_control = n * (n - 1) * (
                control_points[2:] - 2 * control_points[1:-1] + control_points[:-2]
            )
            second_deriv = self._eval_bezier(second_deriv_control, s)
            return first_deriv, second_deriv

        return (first_deriv,)

    def _eval_bezier(self, control_points: np.ndarray, s: float) -> np.ndarray:
        """
        计算贝塞尔曲线在参数s处的值

        使用de Casteljau算法进行稳定计算

        Args:
            control_points: 控制点数组
            s: 参数值

        Returns:
            曲线点
        """
        points = control_points.copy()
        n = len(points)

        # de Casteljau算法
        for k in range(1, n):
            for i in range(n - k):
                points[i] = (1 - s) * points[i] + s * points[i + 1]

        return points[0]

    def _compute_curvature_at_point(
        self,
        control_points: np.ndarray,
        s: float
    ) -> Tuple[float, float]:
        """
        计算参数s处的曲率和速度模长

        曲率公式：κ = (ẋÿ - ẏẍ) / (ẋ² + ẏ²)^(3/2)

        Args:
            control_points: 控制点数组，形状(n, 2)
            s: 参数值，范围[0, 1]

        Returns:
            (curvature, speed): 曲率和速度模长
        """
        # 计算一阶和二阶导数
        first_deriv, second_deriv = self._compute_bezier_derivatives(
            control_points, s, order=2
        )

        x_dot = first_deriv[0]
        y_dot = first_deriv[1]
        x_ddot = second_deriv[0]
        y_ddot = second_deriv[1]

        # 使用安全的曲率计算函数，避免除零
        curvature, speed = safe_curvature_calculation(
            x_dot, y_dot, x_ddot, y_ddot,
            epsilon=self.config.numerical_tolerance
        )

        return curvature, speed

    def compute_curvature_squared_cost(
        self,
        control_points: np.ndarray
    ) -> float:
        """
        计算曲率平方积分成本

        J = ∫κ²(s)ds

        使用数值积分计算

        Args:
            control_points: 贝塞尔曲线控制点，形状(n, 2)

        Returns:
            成本值
        """
        cost = 0.0

        for s_i, w_i in zip(self.sampling_points, self.weights):
            kappa, speed = self._compute_curvature_at_point(control_points, s_i)
            # 曲率平方积分：∫κ²(s) * ||r'(s)|| ds
            cost += w_i * kappa**2 * speed

        return cost

    def compute_curvature_derivative_cost(
        self,
        control_points: np.ndarray
    ) -> float:
        """
        计算曲率导数平方积分成本

        J = ∫(dκ/ds)²ds

        使用有限差分计算曲率导数

        Args:
            control_points: 贝塞尔曲线控制点，形状(n, 2)

        Returns:
            成本值
        """
        # 计算采样点的曲率
        curvatures = []
        for s in self.sampling_points:
            kappa, _ = self._compute_curvature_at_point(control_points, s)
            curvatures.append(kappa)

        curvatures = np.array(curvatures)

        # 使用有限差分计算曲率导数
        # dκ/ds ≈ (κ_{i+1} - κ_{i-1}) / (2 * Δs)
        ds = self.sampling_points[1] - self.sampling_points[0] if len(self.sampling_points) > 1 else 1.0

        curvature_derivatives = np.gradient(curvatures, ds)

        # 计算积分：∫(dκ/ds)² * ||r'(s)|| ds
        cost = 0.0
        for i, (s_i, w_i) in enumerate(zip(self.sampling_points, self.weights)):
            _, speed = self._compute_curvature_at_point(control_points, s_i)
            cost += w_i * curvature_derivatives[i]**2 * speed

        return cost

    def compute_curvature_peak_cost(
        self,
        control_points: np.ndarray
    ) -> float:
        """
        计算曲率峰值惩罚成本

        J = max(κ²(s))

        在多个采样点中寻找最大曲率

        Args:
            control_points: 贝塞尔曲线控制点，形状(n, 2)

        Returns:
            成本值
        """
        max_kappa_squared = 0.0

        for s_i in self.sampling_points:
            kappa, _ = self._compute_curvature_at_point(control_points, s_i)
            max_kappa_squared = max(max_kappa_squared, kappa**2)

        return max_kappa_squared

    def add_curvature_cost_to_gcs(
        self,
        bezier_gcs,
        weights: CurvatureCostWeights,
        verbose: bool = True
    ) -> None:
        """
        将曲率成本添加到GCS优化问题

        根据权重配置，添加相应的成本项：
        - curvature_squared: 曲率平方积分成本
        - curvature_derivative: 曲率导数平方积分成本
        - curvature_peak: 曲率峰值惩罚成本

        Args:
            bezier_gcs: BezierGCS对象
            weights: 成本权重配置
            verbose: 是否输出调试信息
        """
        if not weights.is_enabled():
            if verbose:
                print("[CurvatureCost] No curvature cost enabled.")
            return

        if verbose:
            print("[CurvatureCost] Adding curvature penalty costs to GCS...")
            if weights.curvature_squared > 0:
                print(f"  Curvature squared weight: {weights.curvature_squared}")
            if weights.curvature_derivative > 0:
                print(f"  Curvature derivative weight: {weights.curvature_derivative}")
            if weights.curvature_peak > 0:
                print(f"  Curvature peak weight: {weights.curvature_peak}")

        # 添加曲率平方积分成本
        if weights.curvature_squared > 0:
            if self.config.enable_convex_relaxation:
                self._add_convex_relaxed_curvature_cost(
                    bezier_gcs,
                    weights.curvature_squared,
                    verbose
                )
            else:
                # 添加非凸成本（需要SCP处理）
                if verbose:
                    print("  [Warning] Non-convex curvature cost requires SCP handling.")
                self._add_nonconvex_curvature_cost(
                    bezier_gcs,
                    weights.curvature_squared,
                    verbose
                )

        # 添加曲率导数平方积分成本
        if weights.curvature_derivative > 0:
            success = self.add_curvature_derivative_cost_to_gcs(
                bezier_gcs,
                weights.curvature_derivative,
                verbose
            )
            if not success and verbose:
                print("  [Warning] Failed to add curvature derivative cost.")

        # 添加曲率峰值惩罚成本
        if weights.curvature_peak > 0:
            success = self.add_curvature_peak_cost_to_gcs(
                bezier_gcs,
                weights.curvature_peak,
                verbose=verbose
            )
            if not success and verbose:
                print("  [Warning] Failed to add curvature peak cost.")

    def _add_convex_relaxed_curvature_cost(
        self,
        bezier_gcs,
        weight: float,
        verbose: bool,
        edges=None,
    ) -> None:
        """
        使用凸松弛方法添加曲率成本

        松弛方法：速度加权二阶导数平方
        J_relaxed = ∫||r''(s)||² ds

        这是凸的，因为r''(s)是控制点的线性函数

        Args:
            bezier_gcs: BezierGCS对象
            weight: 权重
            verbose: 是否输出调试信息
            edges: 指定添加成本的边列表（方案C分段差异化）。
                若为None，则对所有非source边添加。
        """
        if verbose:
            edge_info = f" ({len(edges)} edges)" if edges is not None else " (all edges)"
            print(f"  Adding convex relaxed curvature cost (weight={weight}){edge_info}...")
            print(f"    Method: speed-weighted second derivative squared")
            print(f"    J_relaxed = ∫||r''(s)||² ds")

        try:
            if hasattr(bezier_gcs, 'addPathEnergyCost'):
                bezier_gcs.addPathEnergyCost(weight, edges=edges)
            elif hasattr(bezier_gcs, 'add_derivative_cost'):
                bezier_gcs.add_derivative_cost(order=2, weight=weight)
            else:
                if verbose:
                    print("  [Warning] No suitable method found in BezierGCS for curvature cost.")
                    print("  Please implement addPathEnergyCost or add_derivative_cost.")
        except Exception as e:
            if verbose:
                print(f"  [Error] Failed to add convex relaxed curvature cost: {e}")


    def compute_cost_gradient(
        self,
        control_points: np.ndarray,
        weights: CurvatureCostWeights
    ) -> np.ndarray:
        """
        计算成本对控制点的梯度

        Args:
            control_points: 贝塞尔曲线控制点，形状(n, 2)
            weights: 成本权重配置

        Returns:
            梯度 ∂J/∂P，形状(n, 2)
        """
        n = len(control_points)
        gradient = np.zeros((n, 2))

        # 使用数值梯度（有限差分）
        epsilon = 1e-6

        for i in range(n):
            for j in range(2):
                # 前向差分
                control_points_plus = control_points.copy()
                control_points_plus[i, j] += epsilon

                cost_plus = 0.0
                if weights.curvature_squared > 0:
                    cost_plus += weights.curvature_squared * self.compute_curvature_squared_cost(control_points_plus)
                if weights.curvature_derivative > 0:
                    cost_plus += weights.curvature_derivative * self.compute_curvature_derivative_cost(control_points_plus)
                if weights.curvature_peak > 0:
                    cost_plus += weights.curvature_peak * self.compute_curvature_peak_cost(control_points_plus)

                # 当前成本
                cost_current = 0.0
                if weights.curvature_squared > 0:
                    cost_current += weights.curvature_squared * self.compute_curvature_squared_cost(control_points)
                if weights.curvature_derivative > 0:
                    cost_current += weights.curvature_derivative * self.compute_curvature_derivative_cost(control_points)
                if weights.curvature_peak > 0:
                    cost_current += weights.curvature_peak * self.compute_curvature_peak_cost(control_points)

                gradient[i, j] = (cost_plus - cost_current) / epsilon

        return gradient

    # ==================== 新增方法：SCP线性化支持 ====================

    def linearize_curvature_cost(
        self,
        control_points: np.ndarray,
        weights: CurvatureCostWeights
    ) -> LinearizedCostCoefficients:
        """
        线性化曲率成本（供SCP调用）

        对曲率平方积分成本进行泰勒展开线性化：
        J ≈ J₀ + gᵀΔP + 0.5 * ΔPᵀHΔP

        Args:
            control_points: 当前控制点，形状(n, 2)
            weights: 成本权重配置

        Returns:
            LinearizedCostCoefficients: 线性化系数
        """
        if weights.curvature_squared <= 0:
            # 返回零系数
            n = len(control_points)
            return LinearizedCostCoefficients(
                gradient=np.zeros(n * 2),
                hessian_diag=np.ones(n * 2) * 1e-6,
                constant=0.0
            )

        return self.linearizer.get_linearized_cost_coeffs(
            control_points,
            weight=weights.curvature_squared
        )

    def compute_analytic_gradient(
        self,
        control_points: np.ndarray,
        weights: CurvatureCostWeights
    ) -> np.ndarray:
        """
        计算解析梯度（替代数值差分）

        使用解析公式计算成本对控制点的梯度，效率比数值差分高10倍以上

        Args:
            control_points: 贝塞尔曲线控制点，形状(n, 2)
            weights: 成本权重配置

        Returns:
            梯度 ∂J/∂P，形状(n, 2)
        """
        return self.gradient_calculator.compute_gradient_2d(control_points, weights)

    def add_curvature_derivative_cost_to_gcs(
        self,
        bezier_gcs,
        weight: float,
        verbose: bool = True,
        edges=None,
    ) -> bool:
        """
        将曲率导数成本添加到GCS

        使用代理成本: J = ∫||r'''(s)||² ds

        Args:
            bezier_gcs: BezierGCS对象
            weight: 成本权重
            verbose: 是否输出调试信息
            edges: 指定添加成本的边列表（方案C分段差异化）

        Returns:
            是否成功添加
        """
        return self.derivative_cost.add_to_gcs(bezier_gcs, weight, verbose, edges=edges)

    def add_curvature_peak_cost_to_gcs(
        self,
        bezier_gcs,
        weight: float,
        control_points: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> bool:
        """
        将曲率峰值成本添加到GCS

        使用松弛方法: min κ_max s.t. |κ(s_i)| ≤ κ_max

        Args:
            bezier_gcs: BezierGCS对象
            weight: 成本权重
            control_points: 当前控制点（用于线性化）
            verbose: 是否输出调试信息

        Returns:
            是否成功添加
        """
        return self.peak_cost.add_to_gcs(bezier_gcs, weight, control_points, verbose)

    def compute_curvature_and_derivatives(
        self,
        control_points: np.ndarray,
        s: float
    ):
        """
        计算曲率及各阶导数

        Args:
            control_points: 控制点数组，形状(n, 2)
            s: 参数值，范围[0, 1]

        Returns:
            CurvatureDerivatives: 曲率及各阶导数
        """
        return self.linearizer.compute_curvature_and_derivatives(control_points, s)

    def clear_gradient_cache(self) -> None:
        """清除梯度计算缓存"""
        self.gradient_calculator.clear_cache()

    def add_segmented_curvature_cost_to_gcs(
        self,
        bezier_gcs,
        boundary_weights: CurvatureCostWeights,
        internal_weights: CurvatureCostWeights,
        boundary_edges: list,
        internal_edges: list,
        transition_edges: Optional[list] = None,
        transition_weights: Optional[CurvatureCostWeights] = None,
        verbose: bool = True,
    ) -> None:
        """将分段差异化曲率成本添加到GCS优化问题（方案C）

        边界段和内部段使用不同的曲率成本权重，过渡段使用插值权重。
        这使得边界段（航向角约束生效处）和内部段可以有不同的曲率优化策略。

        Args:
            bezier_gcs: BezierGCS对象
            boundary_weights: 边界段曲率成本权重
            internal_weights: 内部段曲率成本权重
            boundary_edges: 边界段边列表
            internal_edges: 内部段边列表
            transition_edges: 过渡段边列表（可选）
            transition_weights: 过渡段曲率成本权重（可选，默认取边界和内部的均值）
            verbose: 是否输出调试信息
        """
        if verbose:
            print("[CurvatureCost] Adding segmented curvature costs (Plan C)...")
            print(f"  Boundary edges: {len(boundary_edges)}")
            print(f"  Internal edges: {len(internal_edges)}")
            if transition_edges:
                print(f"  Transition edges: {len(transition_edges)}")

        # 边界段曲率成本
        if boundary_weights.is_enabled() and boundary_edges:
            self._add_segment_curvature_cost(
                bezier_gcs, boundary_weights, boundary_edges,
                "boundary", verbose,
            )

        # 内部段曲率成本
        if internal_weights.is_enabled() and internal_edges:
            self._add_segment_curvature_cost(
                bezier_gcs, internal_weights, internal_edges,
                "internal", verbose,
            )

        # 过渡段曲率成本（使用插值权重）
        if transition_edges:
            if transition_weights is None:
                transition_weights = CurvatureCostWeights(
                    curvature_squared=(
                        boundary_weights.curvature_squared
                        + internal_weights.curvature_squared
                    )
                    / 2,
                    curvature_derivative=(
                        boundary_weights.curvature_derivative
                        + internal_weights.curvature_derivative
                    )
                    / 2,
                    curvature_peak=0.0,
                )
            if transition_weights.is_enabled():
                self._add_segment_curvature_cost(
                    bezier_gcs, transition_weights, transition_edges,
                    "transition", verbose,
                )

    def _add_segment_curvature_cost(
        self,
        bezier_gcs,
        weights: CurvatureCostWeights,
        edges: list,
        segment_name: str,
        verbose: bool,
    ) -> None:
        """为指定段的边添加曲率成本

        Args:
            bezier_gcs: BezierGCS对象
            weights: 成本权重
            edges: 边列表
            segment_name: 段名称（用于日志）
            verbose: 是否输出调试信息
        """
        if verbose:
            print(f"  [{segment_name}] Adding curvature cost to {len(edges)} edges")

        # 曲率平方积分成本
        if weights.curvature_squared > 0:
            if self.config.enable_convex_relaxation:
                self._add_convex_relaxed_curvature_cost(
                    bezier_gcs, weights.curvature_squared, verbose, edges=edges
                )
            else:
                if verbose:
                    print(f"  [{segment_name}] Non-convex curvature cost requires SCP.")

        # 曲率导数平方积分成本
        if weights.curvature_derivative > 0:
            try:
                self.derivative_cost.add_to_gcs(
                    bezier_gcs, weights.curvature_derivative, verbose, edges=edges
                )
            except Exception as e:
                if verbose:
                    print(f"  [{segment_name}] Failed to add derivative cost: {e}")

        # 曲率峰值惩罚成本
        if weights.curvature_peak > 0:
            try:
                self.peak_cost.add_to_gcs(
                    bezier_gcs, weights.curvature_peak, None, verbose
                )
            except Exception as e:
                if verbose:
                    print(f"  [{segment_name}] Failed to add peak cost: {e}")
