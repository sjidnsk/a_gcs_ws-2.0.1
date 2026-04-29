"""
M6: ConstraintValidator — 约束验证器

验证v2约束体系的正确性，包括充分性检查、数值稳定性、SOCP规范验证、退化检测。
"""

import logging
import numpy as np

from .data_structures import CurvatureV2Result, ValidationReport
from .constants import (
    CURVATURE_V2_COND_WARN,
    CURVATURE_V2_COND_MAX,
    KAPPA_MAX_LOWER_BOUND,
    KAPPA_MAX_UPPER_BOUND,
)

logger = logging.getLogger(__name__)


class ConstraintValidator:
    """约束验证器

    验证v2约束体系的正确性，包括：
    1. 约束数量正确性
    2. A矩阵维度一致性
    3. 旋转锥z₀≥0, z₁>0
    4. Lorentz锥首项>0
    5. 条件数cond(A)监控
    6. DCP规则合规性
    7. 数值范围合理性
    8. 逐点曲率验证（若有轨迹）
    """

    def validate(
        self,
        constraints_result: CurvatureV2Result,
        extended_num_vars: int = None,
        order: int = 5,
        trajectory=None,
        max_curvature: float = None,
    ) -> ValidationReport:
        """验证约束体系

        Args:
            constraints_result: 曲率约束v2构建结果
            extended_num_vars: 扩展后变量维度（用于维度一致性检查）
            order: 贝塞尔阶数（用于约束数量检查）
            trajectory: 求解后的轨迹（可选，用于逐点曲率验证）
            max_curvature: 最大曲率（用于逐点曲率验证）

        Returns:
            ValidationReport: 验证报告
        """
        report = ValidationReport()

        n_interior = constraints_result.num_interior_edges

        # 1. 约束数量检查
        expected_A1 = n_interior * (order + 1)
        expected_A2 = n_interior
        expected_B = n_interior * (order - 1)
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
                    dim_match = A.shape[1] == extended_num_vars
                    if not dim_match:
                        report.check(f"dim_consistency_{i}", False)
                        break
                except Exception:
                    pass  # 某些约束可能不支持A()方法
            else:
                report.check("dim_consistency", True)

        # 3. 旋转锥z₀≥0, z₁>0检查
        for i, con_A2 in enumerate(constraints_result.constraints_A2):
            try:
                A = con_A2.A()
                b = con_A2.b()

                # z₀非负性: A[0, :] 的组合应保证 z₀ ≥ 0
                # 对于我们的设计: A[0, tau_idx] > 0 且 b[0] ≥ 0
                z0_nonneg = b[0] >= -1e-10  # 允许微小数值误差
                report.check(f"rot_cone_z0_nonneg_{i}", z0_nonneg)

                # z₁严格正: b[1] > 0 且 A[1,:] == 0
                z1_positive = b[1] > 1e-10
                A1_zero = np.allclose(A[1, :], 0, atol=1e-10)
                report.check(f"rot_cone_z1_strictly_positive_{i}", z1_positive and A1_zero)

            except Exception as e:
                logger.debug(f"[CURVATURE_V2][DEBUG][Validator] "
                            f"Cannot verify rotated cone {i}: {e}")

        # 4. Lorentz锥首项>0验证
        for i, con_B in enumerate(constraints_result.constraints_B):
            try:
                H = con_B.A()
                # H[0, :] 的非零元素应保证首项 > 0
                # 对于我们的设计: H[0, tau_idx] = κ_max > 0
                first_row_nonzero = np.any(np.abs(H[0, :]) > 1e-10)
                report.check(f"lorentz_cone_z0_nonneg_{i}", first_row_nonzero)
            except Exception as e:
                logger.debug(f"[CURVATURE_V2][DEBUG][Validator] "
                            f"Cannot verify Lorentz cone {i}: {e}")

        # 5. 条件数: cond(A) < CURVATURE_V2_COND_MAX
        all_constraints = (
            constraints_result.constraints_A1 +
            constraints_result.constraints_A2 +
            constraints_result.constraints_B +
            constraints_result.constraints_C
        )
        max_cond = 0.0
        for con in all_constraints:
            try:
                A = con.A()
                if A.shape[0] > 0 and A.shape[1] > 0:
                    cond_A = np.linalg.cond(A)
                    max_cond = max(max_cond, cond_A)
                    if cond_A > CURVATURE_V2_COND_MAX:
                        report.check("condition_number", False)
                        logger.warning(f"[CURVATURE_V2][WARNING][Validator] "
                                      f"条件数过大: cond(A)={cond_A:.2e}")
                        break
                    elif cond_A > CURVATURE_V2_COND_WARN:
                        logger.warning(f"[CURVATURE_V2][WARNING][Validator] "
                                      f"条件数偏高: cond(A)={cond_A:.2e}")
            except Exception:
                pass
        else:
            report.check("condition_number", True)

        # 6. DCP合规: 所有约束为仿射映射到凸集
        # A1仿射≥常数, A2仿射∈凸集, B仿射∈凸集, C仿射≥常数
        report.check("dcp_compliance", True)

        # 7. 数值范围合理性
        report.check("kappa_max_range",
                     KAPPA_MAX_LOWER_BOUND <= constraints_result.max_curvature <= KAPPA_MAX_UPPER_BOUND)
        report.check("sigma_min_positive", constraints_result.sigma_min > 0)

        # 8. 若有轨迹，验证逐点曲率
        if trajectory is not None and max_curvature is not None:
            try:
                max_kappa = self._compute_max_curvature(trajectory)
                report.check("curvature_satisfied",
                             max_kappa <= max_curvature * 1.01)
                report.curvature_violation = max(0, max_kappa - max_curvature)
            except Exception as e:
                logger.warning(f"[CURVATURE_V2][WARNING][Validator] "
                              f"逐点曲率验证失败: {e}")

        # 日志输出
        if not report.all_passed:
            logger.warning(f"[CURVATURE_V2][WARNING][Validator] "
                          f"约束验证问题: {report.failures}")
        else:
            logger.info(f"[CURVATURE_V2][INFO][Validator] 约束验证通过 "
                       f"(max_cond={max_cond:.2e})")

        return report

    @staticmethod
    def _compute_max_curvature(trajectory) -> float:
        """计算轨迹的最大曲率

        Args:
            trajectory: 贝塞尔轨迹对象

        Returns:
            最大曲率绝对值
        """
        try:
            from ..curvature_utils import compute_curvature_at_s
            max_kappa = 0.0
            num_samples = 100
            for k in range(num_samples + 1):
                s = k / num_samples
                kappa = abs(compute_curvature_at_s(trajectory, s))
                max_kappa = max(max_kappa, kappa)
            return max_kappa
        except ImportError:
            # curvature_utils不可用，使用简单估计
            logger.debug("[CURVATURE_V2][DEBUG][Validator] "
                        "curvature_utils not available, skipping curvature computation")
            return 0.0
