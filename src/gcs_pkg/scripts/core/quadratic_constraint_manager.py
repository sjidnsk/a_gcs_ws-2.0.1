"""
二次约束管理器模块

提供QuadraticConstraintManager类，用于构造阿克曼转向车辆的二次转向角约束。
"""

import numpy as np
from typing import Optional, Tuple

from pydrake.solvers import QuadraticConstraint
from pydrake.symbolic import DecomposeLinearExpressions


__all__ = ['QuadraticConstraintManager']


class QuadraticConstraintManager:
    """
    二次约束管理器

    负责构造转向角约束的QuadraticConstraint。

    数学推导：
    转向角约束：δ_min ≤ δ ≤ δ_max
    其中 δ = atan(L·θ̇/v)，v = √(ẋ² + ẏ²)

    等价形式：tan(δ_min) ≤ L·θ̇/v ≤ tan(δ_max)

    令 k = tan(δ)，则：
    上界：L·θ̇ ≤ k_max·√(ẋ² + ẏ²)
    下界：k_min·√(ẋ² + ẏ²) ≤ L·θ̇

    二次形式（上界）：
    L²·θ̇² ≤ k_max²·(ẋ² + ẏ²)
    k_max²·ẋ² + k_max²·ẏ² - L²·θ̇² ≥ 0

    QuadraticConstraint形式：lb ≤ 0.5·xᵀ·Q·x + bᵀ·x ≤ ub

    对于上界约束：
    Q = 2·[k_max²·H_xᵀ·H_x + k_max²·H_yᵀ·H_y - L²·H_θᵀ·H_θ]
    b = 0
    lb = 0
    ub = ∞

    对于下界约束：
    Q = 2·[k_min²·H_xᵀ·H_x + k_min²·H_yᵀ·H_y - L²·H_θᵀ·H_θ]
    b = 0
    lb = -∞
    ub = 0

    注意：Hessian矩阵是不定的（非凸约束），使用HessianType.kIndefinite。
    """

    def __init__(self, wheelbase: float, u_vars: np.ndarray):
        """
        初始化二次约束管理器

        Args:
            wheelbase: 车辆轴距L（米）
            u_vars: 决策变量数组

        Raises:
            ValueError: 如果wheelbase <= 0
        """
        if wheelbase <= 0:
            raise ValueError(f"轴距必须为正数，当前值：{wheelbase}")

        self.wheelbase = wheelbase
        self.u_vars = u_vars
        self.n_vars = len(u_vars)

    def create_steering_constraint(
        self,
        x_dot_expr,
        y_dot_expr,
        theta_dot_expr,
        tan_delta_min: float,
        tan_delta_max: float
    ) -> Tuple[Optional[QuadraticConstraint], Optional[QuadraticConstraint]]:
        """
        创建转向角约束（上下界）

        Args:
            x_dot_expr: ẋ的符号表达式
            y_dot_expr: ẏ的符号表达式
            theta_dot_expr: θ̇的符号表达式
            tan_delta_min: tan(δ_min)
            tan_delta_max: tan(δ_max)

        Returns:
            (con_upper, con_lower)元组，如果创建失败则返回(None, None)

        Raises:
            ValueError: 如果参数无效
        """
        try:
            # 分解线性表达式
            H_x = DecomposeLinearExpressions([x_dot_expr], self.u_vars)
            H_y = DecomposeLinearExpressions([y_dot_expr], self.u_vars)
            H_theta = DecomposeLinearExpressions([theta_dot_expr], self.u_vars)

            # 构造上界约束的Hessian矩阵
            Q_upper = self._construct_hessian(tan_delta_max, H_x, H_y, H_theta)

            # 构造下界约束的Hessian矩阵
            Q_lower = self._construct_hessian(tan_delta_min, H_x, H_y, H_theta)

            # 线性项
            b = np.zeros(self.n_vars)

            # 创建上界约束：k_max²·ẋ² + k_max²·ẏ² - L²·θ̇² ≥ 0
            con_upper = QuadraticConstraint(
                Q_upper,
                b,
                0.0,  # lb
                np.inf,  # ub
                hessian_type=QuadraticConstraint.HessianType.kIndefinite
            )

            # 创建下界约束：k_min²·ẋ² + k_min²·ẏ² - L²·θ̇² ≤ 0
            con_lower = QuadraticConstraint(
                Q_lower,
                b,
                -np.inf,  # lb
                0.0,  # ub
                hessian_type=QuadraticConstraint.HessianType.kIndefinite
            )

            return con_upper, con_lower

        except Exception as e:
            print(f"创建二次约束失败：{e}")
            return None, None

    def _construct_hessian(
        self,
        k: float,
        H_x: np.ndarray,
        H_y: np.ndarray,
        H_theta: np.ndarray
    ) -> np.ndarray:
        """
        构造Hessian矩阵

        Q = 2·(k²·H_xᵀ·H_x + k²·H_yᵀ·H_y - L²·H_θᵀ·H_θ)

        Args:
            k: tan(δ_max)或tan(δ_min)
            H_x: ẋ的线性表达式系数矩阵
            H_y: ẏ的线性表达式系数矩阵
            H_theta: θ̇的线性表达式系数矩阵

        Returns:
            Hessian矩阵Q（n×n）
        """
        Q = 2 * (k**2 * H_x.T @ H_x + k**2 * H_y.T @ H_y - self.wheelbase**2 * H_theta.T @ H_theta)
        return Q
