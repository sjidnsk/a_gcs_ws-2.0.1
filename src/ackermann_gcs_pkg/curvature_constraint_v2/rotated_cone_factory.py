"""
M4: RotatedConeFactory — 旋转锥约束工厂

封装Drake API的LorentzConeConstraint和RotatedLorentzConeConstraint构造，
提供统一的约束创建接口，含维度验证和DCP合规性检查。
"""

import logging
import numpy as np

from pydrake.solvers import (
    LorentzConeConstraint,
    RotatedLorentzConeConstraint,
)
from pydrake.symbolic import DecomposeLinearExpressions

from .exceptions import ConstraintConstructionError

logger = logging.getLogger(__name__)


class RotatedConeFactory:
    """旋转锥约束工厂

    提供静态方法创建Drake API的LorentzConeConstraint和
    RotatedLorentzConeConstraint，含维度验证。
    """

    @staticmethod
    def create_lorentz_cone(A: np.ndarray, b: np.ndarray,
                            description: str = "") -> LorentzConeConstraint:
        """创建Lorentz锥约束: A @ x + b ∈ LorentzCone

        即: z[0] >= sqrt(z[1]^2 + ... + z[n]^2)

        Args:
            A: 约束矩阵，形状(cone_dim, num_vars)，cone_dim >= 2
            b: 常数向量，形状(cone_dim,)
            description: 约束描述（用于日志）

        Returns:
            LorentzConeConstraint对象

        Raises:
            ConstraintConstructionError: 维度验证失败
        """
        if A.shape[0] < 2:
            raise ConstraintConstructionError(
                f"Lorentz cone requires at least 2D, got A.shape={A.shape}"
            )
        if A.shape[0] != len(b):
            raise ConstraintConstructionError(
                f"A and b dimension mismatch: A.shape[0]={A.shape[0]}, "
                f"len(b)={len(b)}"
            )

        try:
            con = LorentzConeConstraint(A, b)
        except Exception as e:
            raise ConstraintConstructionError(
                f"Failed to create LorentzConeConstraint: {e}"
            ) from e

        if description:
            logger.debug(f"[CURVATURE_V2][DEBUG][Factory] Created Lorentz cone: {description}, "
                        f"dim={A.shape[0]}, vars={A.shape[1]}")

        return con

    @staticmethod
    def create_rotated_lorentz_cone(A: np.ndarray, b: np.ndarray,
                                    description: str = "") -> RotatedLorentzConeConstraint:
        """创建旋转二阶锥约束: A @ x + b ∈ RotatedLorentzCone

        即: z[0]*z[1] >= z[2]^2 + ... + z[n]^2, z[0]>=0, z[1]>=0

        Args:
            A: 约束矩阵，形状(cone_dim, num_vars)，cone_dim >= 3
            b: 常数向量，形状(cone_dim,)
            description: 约束描述（用于日志）

        Returns:
            RotatedLorentzConeConstraint对象

        Raises:
            ConstraintConstructionError: 维度验证失败
        """
        if A.shape[0] < 3:
            raise ConstraintConstructionError(
                f"Rotated Lorentz cone requires at least 3D, got A.shape={A.shape}"
            )
        if A.shape[0] != len(b):
            raise ConstraintConstructionError(
                f"A and b dimension mismatch: A.shape[0]={A.shape[0]}, "
                f"len(b)={len(b)}"
            )

        try:
            con = RotatedLorentzConeConstraint(A, b)
        except Exception as e:
            raise ConstraintConstructionError(
                f"Failed to create RotatedLorentzConeConstraint: {e}"
            ) from e

        if description:
            logger.debug(f"[CURVATURE_V2][DEBUG][Factory] Created rotated Lorentz cone: {description}, "
                        f"dim={A.shape[0]}, vars={A.shape[1]}")

        return con

    @staticmethod
    def create_rotated_cone_for_sigma_tau(
        tau_idx: int, sigma_idx: int, num_vars: int
    ) -> tuple:
        """构建约束A2: (τ_e, 1, σ_e) ∈ RotatedLorentzCone

        即 τ_e * 1 ≥ σ_e²

        Args:
            tau_idx: τ_e在u_vars中的索引
            sigma_idx: σ_e在u_vars中的索引
            num_vars: 变量总数

        Returns:
            (A, b) 元组，用于RotatedLorentzConeConstraint
            A: 形状(3, num_vars)
            b: 形状(3,)
        """
        A = np.zeros((3, num_vars))
        A[0, tau_idx] = 1.0    # z[0] = τ_e
        A[2, sigma_idx] = 1.0  # z[2] = σ_e
        b = np.array([0.0, 1.0, 0.0])  # z[1] = 1

        return A, b

    @staticmethod
    def create_curvature_velocity_coupling(
        Qj_expr, max_curvature: float, tau_idx: int, u_vars
    ) -> tuple:
        """构建约束B: (κ_max·τ_e, Q_j,x, Q_j,y) ∈ LorentzCone

        即 κ_max · τ_e ≥ ‖Q_j‖_2

        Args:
            Qj_expr: 二阶导数控制点的Drake Expression，形状(dimension,)
            max_curvature: 最大曲率κ_max
            tau_idx: τ_e在u_vars中的索引
            u_vars: 决策变量列表

        Returns:
            (H, b) 元组，用于LorentzConeConstraint
            H: 形状(1+dimension, num_vars)
            b: 形状(1+dimension,)
        """
        # 分解Q_j的线性表达式
        A_Qj = DecomposeLinearExpressions(Qj_expr, u_vars)
        num_vars = A_Qj.shape[1]
        dimension = A_Qj.shape[0]

        # 构建 Lorentz 锥: (κ_max·τ_e, Q_j,x, Q_j,y) ∈ LorentzCone
        H = np.zeros((1 + dimension, num_vars))
        H[0, tau_idx] = max_curvature  # z[0] = κ_max * τ_e
        H[1:, :] = A_Qj               # z[1:] = Q_j
        b = np.zeros(1 + dimension)

        return H, b

    @staticmethod
    def create_velocity_lower_bound_linear(
        qi_expr, d_theta: np.ndarray, sigma_idx: int, u_vars
    ) -> tuple:
        """构建约束A1: q_i · d_θ ≥ σ_e (线性不等式)

        Args:
            qi_expr: 一阶导数控制点的Drake Expression，形状(dimension,)
            d_theta: 航向角方向单位向量，形状(dimension,)
            sigma_idx: σ_e在u_vars中的索引
            u_vars: 决策变量列表

        Returns:
            (A_full, lb, ub) 元组，用于LinearConstraint
            A_full: 形状(1, num_vars)
            lb: 形状(1,) 下界
            ub: 形状(1,) 上界
        """
        # q_i · d_θ = qi_x * cos_θ + qi_y * sin_θ
        dot_expr = qi_expr[0] * d_theta[0] + qi_expr[1] * d_theta[1]

        # 分解为线性表达式: A_dot @ x + b_dot
        A_dot = DecomposeLinearExpressions([dot_expr], u_vars)

        # 约束: A_dot @ x - σ_e ≥ 0
        A_full = A_dot.copy()
        A_full[0, sigma_idx] -= 1.0  # 减去 σ_e

        return A_full, np.array([0.0]), np.array([np.inf])
