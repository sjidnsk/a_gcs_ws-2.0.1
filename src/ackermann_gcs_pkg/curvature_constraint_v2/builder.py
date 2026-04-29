"""
M1: CurvatureConstraintV2Builder — 曲率约束v2构建器

对每条GCS内部边，构建完整的v2约束体系（A1+A2+B+C）。

约束体系：
- A1: q_i · d_θ ≥ σ_e          (线性速度下界, n+1个/边)
- A2: τ_e · 1 ≥ σ_e²           (旋转二阶锥, 1个/边)
- B:  κ_max · τ_e ≥ ‖Q_j‖_2   (Lorentz锥, n-1个/边)
- C:  σ_e ≥ σ_min              (下界保证, 1个/边)
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
from .exceptions import (
    InvalidParameterError,
    ConstraintConstructionError,
)

logger = logging.getLogger(__name__)


class CurvatureConstraintV2Builder:
    """曲率约束v2构建器

    对每条GCS内部边，构建旋转二阶锥+线性速度下界的完整约束体系。
    """

    def build(
        self,
        max_curvature: float,
        u_path_dot,
        u_path_ddot,
        heading_directions: dict,
        aux_vars,
        u_vars,
        gcs_edges,
        source_vertex,
        boundary_edge_ids: set,
        sigma_min: float,
    ) -> CurvatureV2Result:
        """构建v2曲率约束体系

        Args:
            max_curvature: κ_max (1/m), >0
            u_path_dot: 一阶导数贝塞尔曲线控制点 (Drake Expression列表)
            u_path_ddot: 二阶导数贝塞尔曲线控制点 (Drake Expression列表)
            heading_directions: 逐边航向角方向 {edge_id: np.array([cos_θ, sin_θ])}
            aux_vars: 辅助变量管理器 (AuxiliaryVarManager实例)
            u_vars: 扩展后的决策变量列表
            gcs_edges: GCS图的边列表
            source_vertex: 源点顶点
            boundary_edge_ids: 边界边ID集合
            sigma_min: σ_e的最小下界, >0

        Returns:
            CurvatureV2Result: 包含所有约束和binding的结果

        Raises:
            InvalidParameterError: 参数验证失败
            ConstraintConstructionError: 约束构建失败
        """
        # 参数验证
        if max_curvature <= 0:
            raise InvalidParameterError(
                f"max_curvature必须为正数, got {max_curvature}"
            )
        if sigma_min <= 0:
            raise InvalidParameterError(
                f"sigma_min必须为正数, got {sigma_min}"
            )

        n = len(u_path_dot) - 1       # 贝塞尔阶数
        n_ddot = len(u_path_ddot)     # = n - 1
        num_vars = len(u_vars)
        dimension = u_path_dot[0].shape[0]  # 支持参数化维度(2D/3D)

        constraints_A1 = []  # 线性速度下界
        constraints_A2 = []  # 旋转二阶锥
        constraints_B  = []  # Lorentz锥
        constraints_C  = []  # σ_e下界
        all_bindings   = []

        num_interior_edges = 0

        for edge in gcs_edges:
            # 跳过source边
            if edge.u() == source_vertex:
                continue
            # 跳过边界边
            if id(edge) in boundary_edge_ids:
                continue

            num_interior_edges += 1

            # 获取该边的辅助变量索引
            sigma_idx = aux_vars.get_sigma_index(edge)
            tau_idx = aux_vars.get_tau_index(edge)

            # 获取该边的航向角方向
            d_theta = heading_directions.get(id(edge), np.array([1.0, 0.0]))
            # 确保d_theta是单位向量
            d_norm = np.linalg.norm(d_theta)
            if d_norm > 0:
                d_theta = d_theta / d_norm

            # === A1: 线性速度下界 ===
            # q_i · d_θ ≥ σ_e, 对 i = 0, ..., n
            for i in range(n + 1):
                qi_expr = u_path_dot[i]  # shape: (dimension,)

                try:
                    A_full, lb, ub = RotatedConeFactory.create_velocity_lower_bound_linear(
                        qi_expr, d_theta, sigma_idx, u_vars
                    )
                    con_A1 = LinearConstraint(A_full, lb, ub)
                except Exception as e:
                    raise ConstraintConstructionError(
                        f"A1约束构建失败 (edge={id(edge)}, i={i}): {e}"
                    ) from e

                constraints_A1.append(con_A1)
                binding = edge.AddConstraint(Binding[Constraint](con_A1, edge.xu()))
                all_bindings.append((edge, binding, 'A1'))

            # === A2: 旋转二阶锥 (τ_e ≥ σ_e²) ===
            # (τ_e, 1, σ_e) ∈ RotatedLorentzCone
            try:
                A_rot, b_rot = RotatedConeFactory.create_rotated_cone_for_sigma_tau(
                    tau_idx, sigma_idx, num_vars
                )
                con_A2 = RotatedConeFactory.create_rotated_lorentz_cone(
                    A_rot, b_rot,
                    description=f"A2: edge={id(edge)}, τ≥σ²"
                )
            except Exception as e:
                raise ConstraintConstructionError(
                    f"A2约束构建失败 (edge={id(edge)}): {e}"
                ) from e

            constraints_A2.append(con_A2)
            binding = edge.AddConstraint(Binding[Constraint](con_A2, edge.xu()))
            all_bindings.append((edge, binding, 'A2'))

            # === B: Lorentz锥 (κ_max · τ_e ≥ ‖Q_j‖_2) ===
            # 对 j = 0, ..., n-2
            for j in range(n_ddot):
                Qj_expr = u_path_ddot[j]  # shape: (dimension,)

                try:
                    H, b = RotatedConeFactory.create_curvature_velocity_coupling(
                        Qj_expr, max_curvature, tau_idx, u_vars
                    )
                    con_B = RotatedConeFactory.create_lorentz_cone(
                        H, b,
                        description=f"B: edge={id(edge)}, j={j}"
                    )
                except Exception as e:
                    raise ConstraintConstructionError(
                        f"B约束构建失败 (edge={id(edge)}, j={j}): {e}"
                    ) from e

                constraints_B.append(con_B)
                binding = edge.AddConstraint(Binding[Constraint](con_B, edge.xu()))
                all_bindings.append((edge, binding, 'B'))

            # === C: σ_e下界保证 ===
            # σ_e ≥ σ_min
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

        logger.info(f"[CURVATURE_V2][INFO][Builder] "
                    f"约束统计: {num_interior_edges}条内部边, "
                    f"共 {len(constraints_A1)}个A1 + {len(constraints_A2)}个A2 + "
                    f"{len(constraints_B)}个B + {len(constraints_C)}个C = "
                    f"{result.total_constraints}个约束")

        return result
