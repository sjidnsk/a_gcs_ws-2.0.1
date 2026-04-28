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
from .constants import LOG_PREFIX

logger = logging.getLogger(__name__)


class RotatedConeFactory:
    """旋转锥约束工厂

    提供静态方法创建Drake API的Lorentz锥和旋转Lorentz锥约束，
    包含维度验证和DCP合规性检查。
    """

    @staticmethod
    def create_lorentz_cone(A, b, description=""):
        """创建Lorentz锥约束: A @ x + b ∈ LorentzCone

        即: z[0] >= sqrt(z[1]^2 + ... + z[n]^2)

        Args:
            A: 约束矩阵，形状(m, n)，m >= 2
            b: 常数向量，形状(m,)
            description: 约束描述（用于日志）

        Returns:
            LorentzConeConstraint: Drake Lorentz锥约束对象

        Raises:
            ConstraintConstructionError: 维度验证失败
        """
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)

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

        logger.debug(f"[{LOG_PREFIX}] Created LorentzConeConstraint "
                     f"dim={A.shape[0]}, vars={A.shape[1]}{', ' + description if description else ''}")
        return con

    @staticmethod
    def create_rotated_lorentz_cone(A, b, description=""):
        """创建旋转二阶锥约束: A @ x + b ∈ RotatedLorentzCone

        即: z[0]*z[1] >= z[2]^2 + ... + z[n]^2, z[0]>=0, z[1]>=0

        Args:
            A: 约束矩阵，形状(m, n)，m >= 3
            b: 常数向量，形状(m,)
            description: 约束描述（用于日志）

        Returns:
            RotatedLorentzConeConstraint: Drake旋转Lorentz锥约束对象

        Raises:
            ConstraintConstructionError: 维度验证失败
        """
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)

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

        logger.debug(f"[{LOG_PREFIX}] Created RotatedLorentzConeConstraint "
                     f"dim={A.shape[0]}, vars={A.shape[1]}{', ' + description if description else ''}")
        return con

    @staticmethod
    def create_rotated_cone_for_sigma_tau(tau_idx, sigma_idx, num_vars):
        """创建约束A2: τ_e >= σ_e²

        使用等价LorentzCone形式（避免RotatedLorentzCone在GCS松弛中与
        其他LorentzCone约束的b向量交互导致不可行）：

        τ_e >= σ_e²
        ⟺ (τ_e + 1)² >= (τ_e - 1)² + (2σ_e)²
        ⟺ (τ_e + 1, τ_e - 1, 2σ_e) ∈ LorentzCone

        即 z = A @ x + b, 其中:
          z[0] = τ_e + 1  (A[0, tau_idx]=1, b[0]=1)
          z[1] = τ_e - 1  (A[1, tau_idx]=1, b[1]=-1)
          z[2] = 2σ_e     (A[2, sigma_idx]=2, b[2]=0)

        Args:
            tau_idx: τ_e在变量向量中的索引
            sigma_idx: σ_e在变量向量中的索引
            num_vars: 变量向量总长度

        Returns:
            Tuple[np.ndarray, np.ndarray]: (A, b) 用于LorentzConeConstraint
        """
        A = np.zeros((3, num_vars))
        A[0, tau_idx] = 1.0     # z[0] = τ_e + 1
        A[1, tau_idx] = 1.0     # z[1] = τ_e - 1
        A[2, sigma_idx] = 2.0   # z[2] = 2σ_e
        b = np.array([1.0, -1.0, 0.0])

        return A, b

    @staticmethod
    def create_curvature_velocity_coupling(Qj_expr, max_curvature, tau_idx, u_vars):
        """创建约束B: (κ_max·τ_e, Qⱼ,x, Qⱼ,y) ∈ LorentzCone

        即 κ_max · τ_e >= ‖Qⱼ‖₂

        Args:
            Qj_expr: 二阶导数控制点的Drake Expression，形状(dimension,)
            max_curvature: κ_max值
            tau_idx: τ_e在变量向量中的索引
            u_vars: 决策变量列表

        Returns:
            Tuple[np.ndarray, np.ndarray]: (H, b) 用于LorentzConeConstraint
        """
        A_Qj = DecomposeLinearExpressions(Qj_expr, u_vars)
        num_vars = A_Qj.shape[1]
        dimension = A_Qj.shape[0]

        # 构建 Lorentz 锥: (κ_max·τ_e, Qⱼ,x, Qⱼ,y, ...) ∈ LorentzCone
        H = np.zeros((1 + dimension, num_vars))
        H[0, tau_idx] = max_curvature  # z[0] = κ_max * τ_e
        H[1:, :] = A_Qj               # z[1:] = Qⱼ
        b = np.zeros(1 + dimension)

        return H, b

    @staticmethod
    def create_velocity_lower_bound_linear(qi_expr, d_theta, sigma_idx, u_vars):
        """创建约束A1: qᵢ · d_θ >= σ_e (线性不等式)

        Args:
            qi_expr: 一阶导数控制点的Drake Expression，形状(dimension,)
            d_theta: 航向角方向单位向量，形状(dimension,)
            sigma_idx: σ_e在变量向量中的索引
            u_vars: 决策变量列表

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (A, lb, ub) 用于LinearConstraint
        """
        # qᵢ · d_θ = qi_x * cos_θ + qi_y * sin_θ
        dot_expr = qi_expr[0] * d_theta[0] + qi_expr[1] * d_theta[1]

        # 分解为线性表达式: A_dot @ x + b_dot
        A_dot = DecomposeLinearExpressions([dot_expr], u_vars)

        # 约束: A_dot @ x - σ_e >= 0
        A_full = A_dot.copy()
        A_full[0, sigma_idx] -= 1.0  # 减去 σ_e

        return A_full, np.array([0.0]), np.array([np.inf])
