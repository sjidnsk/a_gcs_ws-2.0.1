"""
M7: CurvatureConstraintCoordinator — 曲率约束协调器

统一入口，协调v1/v2切换、边界段退化处理、σ_e下界保证、与h̄'迭代协调。
"""

import logging
import numpy as np

from .builder import CurvatureConstraintV2Builder
from .auxiliary_var_manager import AuxiliaryVarManager
from .heading_dir_extractor import HeadingDirExtractor
from .solver_adapter import SolverAdapter
from .constraint_validator import ConstraintValidator
from .data_structures import CurvatureV2Result
from .exceptions import (
    InvalidParameterError,
    PrerequisiteViolationError,
    SolverNotSupportedError,
)
from .constants import (
    SIGMA_MIN_DEFAULT, LOG_PREFIX,
    SIGMA_MIN_LOWER_BOUND, SIGMA_MIN_UPPER_BOUND,
)

logger = logging.getLogger(__name__)


class CurvatureConstraintCoordinator:
    """曲率约束协调器

    统一入口：v1/v2切换、边界段退化处理、前提检查。
    """

    def __init__(self, bezier_gcs, ackermann_gcs=None):
        """初始化协调器

        Args:
            bezier_gcs: BezierGCS实例
            ackermann_gcs: AckermannBezierGCS实例（可选，用于航向角信息）
        """
        self.bezier_gcs = bezier_gcs
        self.ackermann_gcs = ackermann_gcs
        self.solver_adapter = SolverAdapter()
        self.aux_var_manager = None
        self.heading_extractor = HeadingDirExtractor()
        self.validator = ConstraintValidator()

    def add_curvature_constraint(self, config):
        """统一入口: 添加曲率约束

        根据配置自动选择v1或v2，处理退化情况

        Args:
            config: 约束配置对象，需包含：
                - curvature_constraint_version: "v1" 或 "v2"
                - max_curvature: κ_max
                - enable_curvature_hard_constraint: 是否启用
                - heading_method: 航向角约束方法
                - sigma_min: σ_e下界
                - solver_type: 求解器类型（可选）

        Returns:
            CurvatureV2Result 或 None: v2结果，v1时返回None
        """
        # Step 1: 前提条件检查
        # P1: 航向角方法 == ROTATION_MATRIX
        heading_method = getattr(config, 'heading_method', 'rotation_matrix')
        if heading_method != 'rotation_matrix':
            logger.warning(f"[{LOG_PREFIX}] 航向角方法非ROTATION_MATRIX，回退v1 (P1)")
            return self._add_v1(config)

        # P4: 求解器支持旋转锥
        solver_type = getattr(config, 'solver_type', None)
        self.solver_adapter = SolverAdapter(solver_type)
        supports_v2 = self.solver_adapter.check_rotated_cone_support()
        if not supports_v2:
            logger.warning(f"[{LOG_PREFIX}] 求解器不支持旋转锥，回退v1 (P4)")
            return self._add_v1(config)

        # P5: κ_max > 0, σ_min > 0
        max_curvature = config.max_curvature
        if max_curvature <= 0:
            raise InvalidParameterError(f"κ_max必须为正数 (P5), got {max_curvature}")

        sigma_min = self._resolve_sigma_min(config)
        if sigma_min <= 0:
            raise InvalidParameterError(f"σ_min必须为正数 (P5), got {sigma_min}")

        # P6: RotatedLorentzConeConstraint可导入
        try:
            from pydrake.solvers import RotatedLorentzConeConstraint  # noqa: F401
        except ImportError:
            logger.warning(f"[{LOG_PREFIX}] RotatedLorentzConeConstraint不可导入，回退v1 (P6)")
            return self._add_v1(config)

        # Step 2: 版本选择
        use_v2 = (getattr(config, 'curvature_constraint_version', 'v1') == "v2"
                  and supports_v2
                  and getattr(config, 'enable_curvature_hard_constraint', True))

        if not use_v2:
            logger.info(f"[{LOG_PREFIX}] Using curvature constraint v1 (Lorentz cone)")
            return self._add_v1(config)

        # Step 3: v2约束添加
        logger.info(f"[{LOG_PREFIX}] Using curvature constraint v2 (Rotated Lorentz cone)")
        return self._add_v2(config, sigma_min)

    def _add_v2(self, config, sigma_min):
        """添加v2曲率约束

        Args:
            config: 约束配置
            sigma_min: 已解析的σ_min值

        Returns:
            CurvatureV2Result
        """
        max_curvature = config.max_curvature

        # 边分类
        boundary_edge_ids = getattr(config, 'boundary_edge_ids', None)
        if boundary_edge_ids is None:
            boundary_edge_ids = set()

        # 辅助变量管理
        self.aux_var_manager = AuxiliaryVarManager(
            self.bezier_gcs.u_vars,
            self.bezier_gcs.gcs.Edges(),
            self.bezier_gcs.source
        )

        # 航向角方向提取
        heading_dirs = {}
        if self.ackermann_gcs is not None:
            edge_classifications = {}
            if hasattr(self.ackermann_gcs, '_classify_edges'):
                classified = self.ackermann_gcs._classify_edges()
                # 转换为HeadingDirExtractor期望的格式（使用ID集合）
                edge_classifications = {
                    'first_real_edge_ids': {id(e) for e in classified.get('first_real_edges', [])},
                    'target_edge_ids': {id(e) for e in classified.get('target_edges', [])},
                    'boundary_edge_ids': {id(e) for e in classified.get('boundary_edge_ids', [])},
                }

            heading_config = type('HeadingConfig', (), {})()
            # 优先使用config中的航向角，其次从ackermann_gcs获取
            if hasattr(config, 'source_heading') and config.source_heading is not None:
                heading_config.source_heading = config.source_heading
            elif hasattr(self.ackermann_gcs, 'source_state'):
                heading_config.source_heading = self.ackermann_gcs.source_state.heading
            if hasattr(config, 'target_heading') and config.target_heading is not None:
                heading_config.target_heading = config.target_heading
            elif hasattr(self.ackermann_gcs, 'target_state'):
                heading_config.target_heading = self.ackermann_gcs.target_state.heading

            heading_dirs = self.heading_extractor.extract_per_edge(
                self.bezier_gcs.gcs.Edges(),
                edge_classifications,
                heading_config,
            )

        # 构建约束
        builder = CurvatureConstraintV2Builder()

        # 获取导数控制点
        u_path_dot = self.bezier_gcs.u_r_trajectory.MakeDerivative(1).control_points()
        u_path_ddot = self.bezier_gcs.u_r_trajectory.MakeDerivative(2).control_points()

        result = builder.build(
            max_curvature=max_curvature,
            u_path_dot=u_path_dot,
            u_path_ddot=u_path_ddot,
            heading_directions=heading_dirs,
            aux_vars=self.aux_var_manager,
            u_vars=self.bezier_gcs.u_vars,
            boundary_edge_ids=boundary_edge_ids,
            sigma_min=sigma_min,
            gcs_edges=self.bezier_gcs.gcs.Edges(),
            source_vertex=self.bezier_gcs.source,
        )

        # 验证
        try:
            validation = self.validator.validate(
                result,
                extended_num_vars=len(self.bezier_gcs.u_vars),
                order=self.bezier_gcs.order,
                max_curvature=max_curvature,
                sigma_min=sigma_min,
            )
            if not validation.all_passed:
                logger.warning(f"[{LOG_PREFIX}] Constraint validation issues: {validation.failures}")
        except Exception as e:
            logger.warning(f"[{LOG_PREFIX}] Constraint validation exception: {e}")

        # 存储约束引用
        self.bezier_gcs.curvature_constraints_v2 = result
        self.bezier_gcs._curvature_v2_bindings = result.all_bindings
        self.bezier_gcs._curvature_v2_added = True

        # 添加σ_e和τ_e正则化成本：鼓励求解器选择较小的辅助变量值
        # σ_e和τ_e无上界约束，不加成本会导致松弛解中这些变量不稳定，
        # 影响舍入质量（诊断发现σ_e可达61000+，远超物理合理范围）
        # 使用二次成本σ_e²+τ_e²，无论正负都惩罚大值
        aux_reg_weight = 0.001  # 小权重，不主导成本但提供正则化
        for edge in self.bezier_gcs.gcs.Edges():
            if edge.u() == self.bezier_gcs.source:
                continue
            xu = edge.xu()
            # σ_e是u_vars的倒数第2个变量，τ_e是最后1个
            sigma_expr = xu[-2]
            tau_expr = xu[-1]
            edge.AddCost(aux_reg_weight * (sigma_expr**2 + tau_expr**2))

        return result

    def _add_v1(self, config):
        """回退到v1: 调用现有addCurvatureHardConstraintForEdges

        Args:
            config: 约束配置

        Returns:
            None (v1不返回CurvatureV2Result)
        """
        max_curvature = config.max_curvature
        min_velocity = getattr(config, 'min_velocity', 1.58)
        h_bar_prime = getattr(config, 'h_bar_prime', None)
        h_bar_prime_safety_factor = getattr(config, 'h_bar_prime_safety_factor', 0.7)
        boundary_edge_ids = getattr(config, 'boundary_edge_ids', None)

        self.bezier_gcs.addCurvatureHardConstraintForEdges(
            max_curvature=max_curvature,
            min_velocity=min_velocity,
            boundary_edge_ids=boundary_edge_ids,
            h_bar_prime=h_bar_prime,
            h_bar_prime_safety_factor=h_bar_prime_safety_factor,
        )
        return None

    @staticmethod
    def _resolve_sigma_min(config):
        """解析σ_min值

        Args:
            config: 约束配置

        Returns:
            float: σ_min值
        """
        sigma_min = getattr(config, 'sigma_min', 'auto')

        if sigma_min == "auto":
            # 自动推导: σ_min = order · hdot_min · v_min_physical · 2.0
            order = getattr(config, 'order', 5)
            hdot_min = getattr(config, 'hdot_min', 0.01)
            v_min_physical = getattr(config, 'v_min_physical', 0.1)
            sigma_min = order * hdot_min * v_min_physical * 2.0
            # 限制在合理范围
            sigma_min = max(SIGMA_MIN_LOWER_BOUND, min(sigma_min, SIGMA_MIN_UPPER_BOUND))
            logger.info(f"[{LOG_PREFIX}] σ_min auto-derived: {sigma_min:.6f}")

        return float(sigma_min)
