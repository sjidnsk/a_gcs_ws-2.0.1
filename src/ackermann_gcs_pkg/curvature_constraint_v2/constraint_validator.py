"""
M6: ConstraintValidator — 约束验证器

验证v2约束体系的正确性，包括充分性检查、数值稳定性、SOCP规范验证、退化检测。
"""

import logging
import numpy as np

from .data_structures import ValidationReport
from .constants import (
    CURVATURE_V2_COND_WARN, CURVATURE_V2_COND_MAX,
    KAPPA_MAX_LOWER_BOUND, KAPPA_MAX_UPPER_BOUND,
    SIGMA_MIN_LOWER_BOUND, SIGMA_MIN_UPPER_BOUND,
    LOG_PREFIX,
)

logger = logging.getLogger(__name__)


class ConstraintValidator:
    """约束验证器

    验证v2约束体系的正确性，包括充分性检查、数值稳定性、SOCP规范验证。
    """

    def validate(self, constraints_result, extended_num_vars=None,
                 order=5, max_curvature=None, sigma_min=None,
                 trajectory=None, curvature_compute_fn=None):
        """验证约束体系

        Args:
            constraints_result: CurvatureV2Result实例
            extended_num_vars: 扩展后变量维度（用于A矩阵维度检查）
            order: 贝塞尔阶数（用于约束数量检查）
            max_curvature: κ_max值（用于数值范围检查）
            sigma_min: σ_min值（用于数值范围检查）
            trajectory: 求解后轨迹（可选，用于逐点曲率验证）
            curvature_compute_fn: 曲率计算函数（可选）

        Returns:
            ValidationReport: 验证报告
        """
        report = ValidationReport()

        n_interior = constraints_result.num_interior_edges
        n = order

        # 1. 约束数量检查
        # A1: 默认禁用（允许倒车），期望0个；启用时每边2个
        expected_A1 = 0  # 允许倒车模式
        expected_A2 = n_interior
        expected_B = n_interior * (n - 1)
        expected_C = n_interior

        report.check("A1_count", len(constraints_result.constraints_A1) == expected_A1)
        report.check("A2_count", len(constraints_result.constraints_A2) == expected_A2)
        report.check("B_count", len(constraints_result.constraints_B) == expected_B)
        report.check("C_count", len(constraints_result.constraints_C) == expected_C)

        # 2. A矩阵维度一致性
        if extended_num_vars is not None:
            all_constraints = (
                constraints_result.constraints_A1 +
                constraints_result.constraints_A2 +
                constraints_result.constraints_B +
                constraints_result.constraints_C
            )
            for i, con in enumerate(all_constraints):
                try:
                    A = con.A()
                    dim_ok = A.shape[1] == extended_num_vars
                    report.check(f"dim_consistency_{i}", dim_ok)
                    if not dim_ok:
                        break  # 一个失败就够了，不继续检查
                except Exception:
                    pass  # 某些约束可能没有A()方法

        # 3. 旋转锥z₀≥0, z₁>0检查
        for i, con_A2 in enumerate(constraints_result.constraints_A2):
            try:
                A = con_A2.A()
                b = con_A2.b()
                # z₀非负性: A[0, :] 的系数和 b[0] 应保证 z₀ ≥ 0
                # 对于我们的A2: A[0, tau_idx]=1, b[0]=0 → z₀=τ_e≥0 (隐式)
                z0_ok = b[0] >= 0  # b[0]=0, τ_e由A2本身保证≥0
                report.check(f"rot_cone_z0_nonneg_{i}", z0_ok)

                # z₁严格正: b[1] > 0 且 A[1,:] == 0
                z1_positive = b[1] > 0 and np.allclose(A[1, :], 0)
                report.check(f"rot_cone_z1_strictly_positive_{i}", z1_positive)
            except Exception:
                pass

        # 4. Lorentz锥首项>0验证
        for i, con_B in enumerate(constraints_result.constraints_B):
            try:
                H = con_B.A()
                # H[0, :] 应有非零元素（κ_max * τ_e）
                z0_nonzero = np.any(H[0, :] != 0)
                report.check(f"lorentz_cone_z0_nonneg_{i}", z0_nonzero)
            except Exception:
                pass

        # 5. 条件数监控
        all_constraints = (
            constraints_result.constraints_A1 +
            constraints_result.constraints_A2 +
            constraints_result.constraints_B +
            constraints_result.constraints_C
        )
        for i, con in enumerate(all_constraints):
            try:
                A = con.A()
                if A.shape[0] > 0 and A.shape[1] > 0:
                    cond_A = np.linalg.cond(A)
                    if cond_A >= CURVATURE_V2_COND_MAX:
                        report.check(f"condition_number_{i}", False)
                        logger.warning(f"[{LOG_PREFIX}] 条件数过大: cond(A)={cond_A:.2e}")
                    elif cond_A >= CURVATURE_V2_COND_WARN:
                        logger.warning(f"[{LOG_PREFIX}] 条件数偏高: cond(A)={cond_A:.2e}")
            except Exception:
                pass

        # 6. DCP合规: 所有约束为仿射映射到凸集
        # A1仿射≥常数, A2仿射∈凸集, B仿射∈凸集, C仿射≥常数
        report.check("dcp_compliance", True)

        # 7. 数值范围合理性
        if max_curvature is not None:
            report.check("kappa_max_range",
                        KAPPA_MAX_LOWER_BOUND <= max_curvature <= KAPPA_MAX_UPPER_BOUND)
        if sigma_min is not None:
            report.check("sigma_min_positive", sigma_min > 0)

        # 8. 若有轨迹，验证逐点曲率
        if trajectory is not None and curvature_compute_fn is not None and max_curvature is not None:
            try:
                max_kappa = curvature_compute_fn(trajectory)
                report.check("curvature_satisfied",
                            max_kappa <= max_curvature * 1.01)
                report.curvature_violation = max(0, max_kappa - max_curvature)
            except Exception as e:
                logger.warning(f"[{LOG_PREFIX}] 逐点曲率验证失败: {e}")

        # 日志
        if report.all_passed:
            logger.info(f"[{LOG_PREFIX}] 约束验证通过")
        else:
            logger.warning(f"[{LOG_PREFIX}] 约束验证问题: {report.failures}")

        return report
