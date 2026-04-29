"""
M7: CurvatureConstraintCoordinator — 曲率约束协调器

统一入口，协调v1/v2切换、前提检查、退化处理、σ_e下界保证、与h̄'迭代协调。
"""

import logging
import numpy as np

from .builder import CurvatureConstraintV2Builder
from .auxiliary_var_manager import AuxiliaryVarManager
from .heading_dir_extractor import HeadingDirExtractor
from .solver_adapter import SolverAdapter
from .constraint_validator import ConstraintValidator
from .data_structures import CurvatureV2Result
from .constants import (
    SIGMA_MIN_LOWER_BOUND,
    SIGMA_MIN_UPPER_BOUND,
    KAPPA_MAX_LOWER_BOUND,
    KAPPA_MAX_UPPER_BOUND,
    SIGMA_MIN_DEFAULT,
)
from .exceptions import (
    InvalidParameterError,
    PrerequisiteViolationError,
    ConstraintConstructionError,
    VertexExtensionError,
)

logger = logging.getLogger(__name__)


class CurvatureConstraintCoordinator:
    """曲率约束协调器

    统一入口，协调v1/v2切换、边界段退化处理、σ_e下界保证、与h̄'迭代协调。

    使用方式：
        coordinator = CurvatureConstraintCoordinator(bezier_gcs, ackermann_gcs)
        result = coordinator.add_curvature_constraint(config)
    """

    def __init__(self, bezier_gcs, ackermann_gcs=None):
        """初始化协调器

        Args:
            bezier_gcs: BezierGCS实例
            ackermann_gcs: AckermannBezierGCS实例（可选）
        """
        self.bezier_gcs = bezier_gcs
        self.ackermann_gcs = ackermann_gcs
        self.solver_adapter = SolverAdapter()
        self.aux_var_manager = None
        self.heading_extractor = HeadingDirExtractor()
        self.validator = ConstraintValidator()

    def add_curvature_constraint(self, config) -> CurvatureV2Result:
        """统一入口: 添加曲率约束

        根据配置自动选择v1或v2，处理退化情况。

        Args:
            config: 约束配置对象，需包含：
                - curvature_constraint_version: "v1" 或 "v2"
                - max_curvature: κ_max
                - enable_curvature_hard_constraint: 是否启用
                - heading_method: 航向角约束方法
                - sigma_min: σ_e最小下界
                - solver_type: 求解器类型（可选）

        Returns:
            CurvatureV2Result（v2模式）或None（v1模式，由v1方法直接添加）

        Raises:
            InvalidParameterError: 参数验证失败
        """
        # 提取配置参数
        version = getattr(config, 'curvature_constraint_version', 'v1')
        max_curvature = getattr(config, 'max_curvature', 0.0)
        enable = getattr(config, 'enable_curvature_hard_constraint', False)
        heading_method = getattr(config, 'heading_method', None)
        sigma_min = getattr(config, 'sigma_min', 'auto')
        solver_type = getattr(config, 'solver_type', None)

        # Step 1: 前提条件检查
        # P1: 航向角方法必须为ROTATION_MATRIX
        if heading_method is not None and str(heading_method) != 'ROTATION_MATRIX' and str(heading_method) != 'HeadingConstraintMethod.ROTATION_MATRIX':
            logger.warning("[CURVATURE_V2][WARNING][Coordinator] "
                          "航向角方法非ROTATION_MATRIX，回退v1 (P1)")
            return self._add_v1(config)

        # P2: 航向角覆盖范围检查
        # A1约束q_i·d_θ ≥ σ_e在航向角变化过大时可能不够准确
        source_heading = getattr(config, 'source_heading', None)
        target_heading = getattr(config, 'target_heading', None)
        if source_heading is not None and target_heading is not None:
            delta_heading = abs((target_heading - source_heading + np.pi) % (2 * np.pi) - np.pi)
            if delta_heading > np.pi:
                logger.warning(f"[CURVATURE_V2][WARNING][Coordinator] "
                              f"航向角覆盖范围>{180}° (Δθ={np.degrees(delta_heading):.1f}°), "
                              f"A1约束保守性可能下降 (P2)")

        # P4: 求解器支持旋转锥
        self.solver_adapter = SolverAdapter(solver_type)
        supports_v2 = self.solver_adapter.check_rotated_cone_support()
        if not supports_v2:
            logger.warning("[CURVATURE_V2][WARNING][Coordinator] "
                          "求解器不支持旋转锥，回退v1 (P4)")
            return self._add_v1(config)

        # P5: κ_max > 0, σ_min > 0
        if max_curvature <= 0:
            raise InvalidParameterError(
                f"κ_max必须为正数 (P5), got {max_curvature}"
            )

        # P6: RotatedLorentzConeConstraint可导入
        try:
            from pydrake.solvers import RotatedLorentzConeConstraint
        except ImportError:
            logger.warning("[CURVATURE_V2][WARNING][Coordinator] "
                          "RotatedLorentzConeConstraint不可导入，回退v1 (P6)")
            return self._add_v1(config)

        # Step 2: 版本选择
        use_v2 = (version == "v2" and supports_v2 and enable)

        if not use_v2:
            logger.info(f"[CURVATURE_V2][INFO][Coordinator] "
                       f"Using curvature constraint v1 (Lorentz cone) "
                       f"[version={version}, supports_v2={supports_v2}, enable={enable}]")
            return self._add_v1(config)

        # Step 3: v2约束添加
        logger.info("[CURVATURE_V2][INFO][Coordinator] "
                   "Using curvature constraint v2 (A1:linear + A2:RotatedLorentzCone + B:LorentzCone + C:σ_lb)")
        return self._add_v2(config)

    def _add_v2(self, config) -> CurvatureV2Result:
        """添加v2曲率约束

        Args:
            config: 约束配置对象

        Returns:
            CurvatureV2Result
        """
        max_curvature = config.max_curvature
        sigma_min = getattr(config, 'sigma_min', 'auto')
        boundary_edge_ids = getattr(config, 'boundary_edge_ids', set())

        # σ_min处理
        sigma_min = self._resolve_sigma_min(sigma_min)

        # 边分类
        if self.ackermann_gcs is not None:
            edge_classes = self.ackermann_gcs._classify_edges()
            boundary_edge_ids = edge_classes.get('boundary_edge_ids', set())
        else:
            edge_classes = {
                'first_real_edge_ids': set(),
                'target_edge_ids': set(),
                'boundary_edge_ids': boundary_edge_ids,
            }

        # 辅助变量管理
        edges_for_aux = list(self.bezier_gcs.gcs.Edges())
        self.aux_var_manager = AuxiliaryVarManager(
            self.bezier_gcs.u_vars,
            edges_for_aux,
            self.bezier_gcs.source
        )
        aux_edge_ids = set(self.aux_var_manager.edge_aux_var_map.keys())

        # u_vars已在BezierGCS.__init__中扩展，直接使用
        extended_u_vars = self.bezier_gcs.u_vars

        # 航向角方向提取
        heading_config = getattr(self.ackermann_gcs, 'heading_constraint_config', None)
        if heading_config is None:
            # 创建默认配置
            heading_config = type('DefaultHeadingConfig', (), {
                'source_heading': 0.0,
                'target_heading': 0.0,
            })()

        u_path_dot = self.bezier_gcs.u_r_trajectory.MakeDerivative(1).control_points()

        heading_dirs = self.heading_extractor.extract_per_edge(
            edges_for_aux,
            edge_classes,
            heading_config,
            u_path_dot,
            extended_u_vars,
        )

        # 构建约束
        u_path_ddot = self.bezier_gcs.u_r_trajectory.MakeDerivative(2).control_points()

        builder = CurvatureConstraintV2Builder()
        try:
            result = builder.build(
                max_curvature=max_curvature,
                u_path_dot=u_path_dot,
                u_path_ddot=u_path_ddot,
                heading_directions=heading_dirs,
                aux_vars=self.aux_var_manager,
                u_vars=extended_u_vars,
                gcs_edges=edges_for_aux,
                source_vertex=self.bezier_gcs.source,
                boundary_edge_ids=boundary_edge_ids,
                sigma_min=sigma_min,
            )
        except Exception as e:
            logger.error(f"[CURVATURE_V2][ERROR][Coordinator] 约束构建失败: {e}")
            raise

        # 验证
        try:
            validation = self.validator.validate(
                result,
                extended_num_vars=self.aux_var_manager.extended_num_vars,
                order=self.bezier_gcs.order,
            )
            if not validation.all_passed:
                logger.warning(f"[CURVATURE_V2][WARNING][Coordinator] "
                              f"约束验证问题: {validation.failures}")
            else:
                logger.info("[CURVATURE_V2][INFO][Coordinator] 约束验证通过")
        except Exception as e:
            logger.warning(f"[CURVATURE_V2][WARNING][Coordinator] 约束验证异常: {e}")

        # τ_e正则化：仅惩罚τ_e，不惩罚σ_e
        # σ_e越大意味着速度越快，曲率约束越紧，不应惩罚
        # τ_e正则化防止τ_e过度松弛（τ_e >> σ_e²），改善保守性
        tau_reg_weight = getattr(config, 'tau_regularization_weight', 0.001)
        if tau_reg_weight > 0:
            num_reg_edges = 0
            for edge in self.bezier_gcs.gcs.Edges():
                if edge.u() == self.bezier_gcs.source:
                    continue
                if id(edge) in boundary_edge_ids:
                    continue
                tau_idx = self.aux_var_manager.get_tau_index(edge)
                xu = edge.xu()
                tau_expr = xu[tau_idx]
                edge.AddCost(tau_reg_weight * tau_expr**2)
                num_reg_edges += 1
            logger.info(f"[CURVATURE_V2][INFO][Coordinator] "
                       f"τ_e正则化已添加: {num_reg_edges}条边, weight={tau_reg_weight}")

        # 存储约束引用
        self.bezier_gcs.curvature_constraints_v2 = result
        self.bezier_gcs._curvature_v2_bindings = result.all_bindings

        return result

    def _add_v1(self, config):
        """回退到v1: 调用现有addCurvatureHardConstraintForEdges

        Args:
            config: 约束配置对象

        Returns:
            None (v1约束直接添加到GCS图中)
        """
        max_curvature = getattr(config, 'max_curvature', 0.0)
        min_velocity = getattr(config, 'min_velocity', 1.0)
        boundary_edge_ids = getattr(config, 'boundary_edge_ids', None)
        h_bar_prime = getattr(config, 'h_bar_prime', None)
        h_bar_prime_safety_factor = getattr(config, 'h_bar_prime_safety_factor', 0.7)

        self.bezier_gcs.addCurvatureHardConstraintForEdges(
            max_curvature=max_curvature,
            min_velocity=min_velocity,
            boundary_edge_ids=boundary_edge_ids,
            h_bar_prime=h_bar_prime,
            h_bar_prime_safety_factor=h_bar_prime_safety_factor,
        )

        return None

    @staticmethod
    def _resolve_sigma_min(sigma_min):
        """解析σ_min参数

        Args:
            sigma_min: "auto"或正float

        Returns:
            解析后的σ_min正值
        """
        if sigma_min == "auto":
            # 自动推导: σ_min = order * hdot_min * v_min_physical * 2.0
            # 典型值: 5 * 0.01 * 0.1 * 2.0 = 0.01
            resolved = SIGMA_MIN_DEFAULT
            logger.info(f"[CURVATURE_V2][INFO][Coordinator] "
                       f"σ_min auto-resolved to {resolved:.6f}")
            return resolved
        else:
            if sigma_min <= 0:
                raise InvalidParameterError(
                    f"sigma_min必须为正数, got {sigma_min}"
                )
            if sigma_min < SIGMA_MIN_LOWER_BOUND:
                logger.warning(f"[CURVATURE_V2][WARNING][Coordinator] "
                              f"sigma_min={sigma_min:.2e} below lower bound "
                              f"{SIGMA_MIN_LOWER_BOUND:.2e}, may cause numerical instability")
            if sigma_min > SIGMA_MIN_UPPER_BOUND:
                logger.warning(f"[CURVATURE_V2][WARNING][Coordinator] "
                              f"sigma_min={sigma_min:.2e} above upper bound "
                              f"{SIGMA_MIN_UPPER_BOUND:.2e}, constraint may be too tight")
            return sigma_min
