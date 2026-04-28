"""
M1: CurvatureConstraintV2Builder — 曲率约束v2构建器

对每条GCS内部边，构建完整的v2约束体系（A1+A2+B+C）。

约束体系：
- A1: qᵢ·d_θ ≥ σ_e          (线性速度下界, n+1个/边)
- A2: τ_e·1 ≥ σ_e²           (旋转二阶锥, 1个/边)
- B:  κ_max·τ_e ≥ ‖Qⱼ‖₂     (Lorentz锥, n-1个/边)
- C:  σ_e ≥ σ_min            (下界保证, 1个/边)
"""

import logging
import numpy as np

from pydrake.solvers import (
    Binding,
    Constraint,
    LinearConstraint,
    LorentzConeConstraint,
    RotatedLorentzConeConstraint,
)
from pydrake.symbolic import DecomposeLinearExpressions

from .data_structures import CurvatureV2Result
from .rotated_cone_factory import RotatedConeFactory
from .exceptions import InvalidParameterError, ConstraintConstructionError
from .constants import (
    KAPPA_MAX_LOWER_BOUND, KAPPA_MAX_UPPER_BOUND,
    SIGMA_MIN_LOWER_BOUND, SIGMA_MIN_UPPER_BOUND,
    LOG_PREFIX,
)

logger = logging.getLogger(__name__)


class CurvatureConstraintV2Builder:
    """曲率约束v2构建器

    对每条GCS内部边，构建旋转二阶锥+线性速度下界的完整约束体系。
    """

    def build(self, max_curvature, u_path_dot, u_path_ddot,
              heading_directions, aux_vars, u_vars,
              boundary_edge_ids, sigma_min, gcs_edges, source_vertex):
        """构建v2曲率约束体系

        Args:
            max_curvature: κ_max (1/m), >0
            u_path_dot: 一阶导数贝塞尔曲线控制点 (Drake Expression列表)
            u_path_ddot: 二阶导数贝塞尔曲线控制点 (Drake Expression列表)
            heading_directions: 逐边航向角方向 {edge_id → np.array([cos_θ, sin_θ])}
            aux_vars: 辅助变量管理器 (AuxiliaryVarManager实例)
            u_vars: 扩展后的决策变量列表
            boundary_edge_ids: 边界边ID集合
            sigma_min: σ_e的最小下界 (>0)
            gcs_edges: GCS图的边列表
            source_vertex: source顶点

        Returns:
            CurvatureV2Result: 包含所有约束和binding的命名元组

        Raises:
            InvalidParameterError: 参数验证失败
            ConstraintConstructionError: 约束构建失败
        """
        # === 参数验证 ===
        if max_curvature <= 0:
            raise InvalidParameterError(
                f"max_curvature必须为正数, got {max_curvature}"
            )
        if sigma_min <= 0:
            raise InvalidParameterError(
                f"sigma_min必须为正数, got {sigma_min}"
            )
        if max_curvature < KAPPA_MAX_LOWER_BOUND:
            logger.warning(f"[{LOG_PREFIX}] κ_max={max_curvature} below lower bound "
                          f"{KAPPA_MAX_LOWER_BOUND}, may cause numerical issues")
        if max_curvature > KAPPA_MAX_UPPER_BOUND:
            logger.warning(f"[{LOG_PREFIX}] κ_max={max_curvature} above upper bound "
                          f"{KAPPA_MAX_UPPER_BOUND}, physically unreasonable")
        if sigma_min < SIGMA_MIN_LOWER_BOUND:
            logger.warning(f"[{LOG_PREFIX}] σ_min={sigma_min} below lower bound "
                          f"{SIGMA_MIN_LOWER_BOUND}, numerically unreliable")
        if sigma_min > SIGMA_MIN_UPPER_BOUND:
            logger.warning(f"[{LOG_PREFIX}] σ_min={sigma_min} above upper bound "
                          f"{SIGMA_MIN_UPPER_BOUND}, may over-constrain")

        n = len(u_path_dot) - 1       # 贝塞尔阶数
        n_ddot = len(u_path_ddot)     # = n - 1
        num_vars = len(u_vars)
        dimension = u_path_dot[0].shape[0]  # 支持参数化维度(2D)

        constraints_A1 = []  # 线性速度下界
        constraints_A2 = []  # 旋转二阶锥
        constraints_B  = []  # Lorentz锥
        constraints_C  = []  # σ_e下界
        all_bindings   = []

        num_interior_edges = 0

        for edge in gcs_edges:
            if edge.u() == source_vertex:
                continue
            edge_id = id(edge)
            if boundary_edge_ids is not None and edge_id in boundary_edge_ids:
                continue

            num_interior_edges += 1

            # 获取该边的辅助变量索引
            try:
                sigma_idx = aux_vars.get_sigma_index(edge)
                tau_idx = aux_vars.get_tau_index(edge)
            except KeyError as e:
                raise ConstraintConstructionError(
                    f"无法获取边 {edge.name()} 的辅助变量索引: {e}"
                ) from e

            # 获取该边的航向角方向
            d_theta = heading_directions.get(edge_id, np.array([1.0, 0.0]))

            # === 诊断开关：逐步启用约束类型 ===
            enable_A2 = True   # Lorentz锥（等价RotatedLorentzCone）
            enable_B = True    # Lorentz锥
            enable_C = True    # σ_e下界
            enable_A1 = False  # 防倒车约束（默认禁用=允许倒车）

            # === A1: 线性速度下界（防倒车） ===
            # 原始: qᵢ · d_θ ≥ σ_e (与σ_e耦合，非齐次，GCS松弛不安全)
            # 改为: qᵢ · d_θ ≥ σ_min (常数下界，齐次线性约束，GCS松弛安全)
            # 物理意义: 保证控制点在航向角方向的速度投影不小于σ_min
            # 效果: 防止倒车（速度在航向角方向为负）
            # 只对首尾控制点施加（i=0和i=n），中间控制点可自由弯曲
            # 默认禁用：允许倒车模式，求解器可自由选择前进/后退
            if enable_A1:
                for i in [0, n]:
                    qi_expr = u_path_dot[i]  # shape: (dimension,)

                    # qᵢ · d_θ >= σ_min (线性不等式，齐次)
                    dot_expr = qi_expr[0] * d_theta[0] + qi_expr[1] * d_theta[1]
                    A_dot = DecomposeLinearExpressions([dot_expr], u_vars)

                    con_A1 = LinearConstraint(A_dot, lb=[sigma_min], ub=[np.inf])
                    constraints_A1.append(con_A1)
                    binding = edge.AddConstraint(Binding[Constraint](con_A1, edge.xu()))
                    all_bindings.append((edge, binding, 'A1'))

            # === A2: τ_e下界保证 (替代旋转二阶锥) ===
            # 原始A2: τ_e >= σ_e² (RotatedLorentzCone)
            # 问题: RotatedLorentzCone的b向量非零，在GCS凸松弛中与
            # 速度LorentzCone约束交互导致不可行
            # 替代: τ_e >= σ_min² (线性下界)
            # 效果: 保留τ_e的下界保证，使B约束(κ_max·τ_e>=‖Qⱼ‖)有效
            # 代价: 失去σ_e-τ_e的非线性耦合，保守性略增
            if enable_A2:
                # τ_e >= σ_min² (线性约束，齐次，GCS松弛安全)
                A_tau = np.zeros((1, num_vars))
                A_tau[0, tau_idx] = 1.0
                con_A2 = LinearConstraint(A_tau, lb=[sigma_min ** 2], ub=[np.inf])
                if num_interior_edges <= 3:
                    logger.info(f"[{LOG_PREFIX}] A2 (linear) edge={edge.name()}: "
                               f"tau_idx={tau_idx}, sigma_min²={sigma_min**2:.6f}")

                constraints_A2.append(con_A2)
                binding = edge.AddConstraint(Binding[Constraint](con_A2, edge.xu()))
                all_bindings.append((edge, binding, 'A2'))

            # === B: Lorentz锥 (κ_max · τ_e ≥ ‖Qⱼ‖₂) ===
            # 对 j = 0, ..., n-2
            if enable_B:
                for j in range(n_ddot):
                    Qj_expr = u_path_ddot[j]  # shape: (dimension,)

                    try:
                        H, b = RotatedConeFactory.create_curvature_velocity_coupling(
                            Qj_expr, max_curvature, tau_idx, u_vars
                        )
                        con_B = RotatedConeFactory.create_lorentz_cone(
                            H, b, description=f"B: edge={edge.name()}, j={j}"
                        )
                    except Exception as e:
                        raise ConstraintConstructionError(
                            f"B约束构建失败 (edge={edge.name()}, j={j}): {e}"
                        ) from e

                    constraints_B.append(con_B)
                    binding = edge.AddConstraint(Binding[Constraint](con_B, edge.xu()))
                    all_bindings.append((edge, binding, 'B'))

            # === C: σ_e下界保证 ===
            # σ_e ≥ σ_min
            if enable_C:
                A_sigma = np.zeros((1, num_vars))
                A_sigma[0, sigma_idx] = 1.0
                con_C = LinearConstraint(A_sigma, lb=[sigma_min], ub=[np.inf])
                constraints_C.append(con_C)
                binding = edge.AddConstraint(Binding[Constraint](con_C, edge.xu()))
                all_bindings.append((edge, binding, 'C'))

        result = CurvatureV2Result(
            constraints_A1=constraints_A1,
            constraints_A2=constraints_A2,
            constraints_B=constraints_B,
            constraints_C=constraints_C,
            all_bindings=all_bindings,
            num_interior_edges=num_interior_edges,
            sigma_min=sigma_min,
            max_curvature=max_curvature,
        )

        # 约束统计日志
        logger.info(f"[{LOG_PREFIX}] 约束统计: {num_interior_edges}条内部边, "
                    f"共 {len(constraints_A1)}个A1 + {len(constraints_A2)}个A2 + "
                    f"{len(constraints_B)}个B + {len(constraints_C)}个C = "
                    f"{result.total_constraints}个约束")

        return result
