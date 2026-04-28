"""
曲率约束v2单元测试

测试旋转二阶锥速度耦合方案的各个子模块：
- M2: AuxiliaryVarManager
- M3: HeadingDirExtractor
- M4: RotatedConeFactory
- M5: SolverAdapter
- M6: ConstraintValidator
- M7: CurvatureConstraintCoordinator
- 数据结构: CurvatureV2Result, ValidationReport, TrajectoryConstraints扩展
"""

import numpy as np
import pytest

# Drake可用性检查
pydrake = pytest.importorskip("pydrake")

from ackermann_gcs_pkg.curvature_constraint_v2 import (
    # 数据结构
    CurvatureV2Result,
    ValidationReport,
    # 异常
    InvalidParameterError,
    SolverNotSupportedError,
    PrerequisiteViolationError,
    ConstraintConstructionError,
    VertexExtensionError,
    # 子模块
    RotatedConeFactory,
    AuxiliaryVarManager,
    HeadingDirExtractor,
    SolverAdapter,
    ConstraintValidator,
    CurvatureConstraintV2Builder,
    CurvatureConstraintCoordinator,
    # 常量
    CURVATURE_V2_FLOAT_TOL,
    CURVATURE_V2_SINGULAR_TOL,
    SIGMA_MIN_LOWER_BOUND,
    SIGMA_MIN_UPPER_BOUND,
    DEFAULT_SIGMA_MIN,
)
from ackermann_gcs_pkg.ackermann_data_structures import TrajectoryConstraints


# ============================================================
# M4: RotatedConeFactory 测试
# ============================================================
class TestRotatedConeFactory:
    """旋转锥约束工厂测试"""

    def test_create_lorentz_cone_basic(self):
        """测试基本Lorentz锥创建"""
        A = np.eye(3)
        b = np.array([1.0, 0.0, 0.0])
        con = RotatedConeFactory.create_lorentz_cone(A, b)
        assert con is not None

    def test_create_lorentz_cone_dim_error(self):
        """测试Lorentz锥维度不足错误"""
        A = np.eye(1)
        b = np.array([1.0])
        with pytest.raises(ConstraintConstructionError):
            RotatedConeFactory.create_lorentz_cone(A, b)

    def test_create_lorentz_cone_mismatch(self):
        """测试A和b维度不匹配"""
        A = np.eye(3)
        b = np.array([1.0, 0.0])
        with pytest.raises(ConstraintConstructionError):
            RotatedConeFactory.create_lorentz_cone(A, b)

    def test_create_rotated_lorentz_cone_basic(self):
        """测试基本旋转锥创建"""
        A = np.eye(3)
        b = np.zeros(3)
        con = RotatedConeFactory.create_rotated_lorentz_cone(A, b)
        assert con is not None

    def test_create_rotated_lorentz_cone_dim_error(self):
        """测试旋转锥维度不足错误"""
        A = np.eye(2)
        b = np.zeros(2)
        with pytest.raises(ConstraintConstructionError):
            RotatedConeFactory.create_rotated_lorentz_cone(A, b)

    def test_create_rotated_cone_for_sigma_tau(self):
        """测试σ_e/τ_e旋转锥构造"""
        tau_idx = 5
        sigma_idx = 4
        num_vars = 6
        A, b = RotatedConeFactory.create_rotated_cone_for_sigma_tau(
            tau_idx, sigma_idx, num_vars
        )
        assert A.shape == (3, 6)
        assert A[0, tau_idx] == 1.0   # z[0] = τ_e
        assert A[2, sigma_idx] == 1.0  # z[2] = σ_e
        assert b[1] == 1.0             # z[1] = 1
        assert b[0] == 0.0
        assert b[2] == 0.0


# ============================================================
# M3: HeadingDirExtractor 测试
# ============================================================
class TestHeadingDirExtractor:
    """航向角方向提取器测试"""

    def test_normalize_direction_unit(self):
        """测试单位向量归一化"""
        d = np.array([1.0, 0.0])
        result = HeadingDirExtractor._normalize_direction(d)
        np.testing.assert_allclose(result, d, atol=1e-10)

    def test_normalize_direction_general(self):
        """测试一般向量归一化"""
        d = np.array([3.0, 4.0])
        result = HeadingDirExtractor._normalize_direction(d)
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-10)
        expected = np.array([0.6, 0.8])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_normalize_direction_zero_vector(self):
        """测试零向量归一化异常"""
        d = np.array([0.0, 0.0])
        with pytest.raises(InvalidParameterError):
            HeadingDirExtractor._normalize_direction(d)

    def test_normalize_direction_tiny_vector(self):
        """测试极小向量归一化异常"""
        d = np.array([1e-10, 1e-10])
        with pytest.raises(InvalidParameterError):
            HeadingDirExtractor._normalize_direction(d)

    def test_interpolate_heading_basic(self):
        """测试基本航向角插值"""
        result = HeadingDirExtractor._interpolate_heading(0.0, np.pi / 2, t=0.5)
        np.testing.assert_allclose(result, np.pi / 4, atol=1e-10)

    def test_interpolate_heading_pi_crossing(self):
        """测试±π跨越插值"""
        # 从170°到-170°（跨越±π）
        src = np.radians(170)
        tgt = np.radians(-170)
        result = HeadingDirExtractor._interpolate_heading(src, tgt, t=0.5)
        # 应该接近180°（π）
        assert abs(result) > np.radians(175)

    def test_extract_per_edge_basic(self):
        """测试逐边航向角方向提取"""
        # 创建模拟边对象
        class MockEdge:
            pass

        edges = [MockEdge(), MockEdge(), MockEdge()]
        edge_classes = {
            'source_edges': [edges[0]],
            'first_real_edges': [edges[1]],
            'target_edges': [],
            'middle_edges': [edges[2]],
        }

        directions = HeadingDirExtractor.extract_per_edge(
            edges, edge_classes,
            source_heading=0.0,
            target_heading=np.pi / 2,
        )

        # 应该有3条边的方向
        assert len(directions) == 3
        # 检查所有方向都是单位向量
        for edge_id, d in directions.items():
            np.testing.assert_allclose(np.linalg.norm(d), 1.0, atol=1e-10)


# ============================================================
# M5: SolverAdapter 测试
# ============================================================
class TestSolverAdapter:
    """求解器适配器测试"""

    def test_default_solver(self):
        """测试默认求解器（假设支持）"""
        adapter = SolverAdapter()
        result = adapter.check_rotated_cone_support()
        assert result is True

    def test_mosek_support(self):
        """测试MOSEK支持"""
        from pydrake.solvers import MosekSolver
        adapter = SolverAdapter(MosekSolver)
        result = adapter.check_rotated_cone_support()
        assert result is True

    def test_gurobi_not_supported(self):
        """测试Gurobi不支持"""
        from pydrake.solvers import GurobiSolver
        adapter = SolverAdapter(GurobiSolver)
        result = adapter.check_rotated_cone_support()
        assert result is False

    def test_fallback_strategy(self):
        """测试回退策略"""
        from pydrake.solvers import MosekSolver
        adapter = SolverAdapter(MosekSolver)
        strategy = adapter.get_fallback_strategy()
        assert strategy is None  # MOSEK支持，无需回退

    def test_gurobi_fallback(self):
        """测试Gurobi回退策略"""
        from pydrake.solvers import GurobiSolver
        adapter = SolverAdapter(GurobiSolver)
        strategy = adapter.get_fallback_strategy()
        assert strategy == 'v1'


# ============================================================
# M6: ConstraintValidator 测试
# ============================================================
class TestConstraintValidator:
    """约束验证器测试"""

    def test_empty_result(self):
        """测试空约束结果验证"""
        result = CurvatureV2Result(sigma_min=DEFAULT_SIGMA_MIN)
        validator = ConstraintValidator()
        report = validator.validate(result, sigma_min=DEFAULT_SIGMA_MIN)
        assert report.all_passed

    def test_validation_report_check(self):
        """测试ValidationReport.check方法"""
        report = ValidationReport()
        report.check("test1", True)
        assert report.all_passed
        assert "test1" not in report.failures

        report.check("test2", False)
        assert not report.all_passed
        assert "test2" in report.failures

    def test_sigma_min_positive(self):
        """测试σ_min正数验证"""
        result = CurvatureV2Result(sigma_min=0.01)
        validator = ConstraintValidator()
        report = validator.validate(result, sigma_min=0.01)
        assert report.checks.get("sigma_min_positive", False)


# ============================================================
# 数据结构测试
# ============================================================
class TestDataStructures:
    """数据结构测试"""

    def test_curvature_v2_result_default(self):
        """测试CurvatureV2Result默认值"""
        result = CurvatureV2Result()
        assert result.constraints_A1 == []
        assert result.constraints_A2 == []
        assert result.constraints_B == []
        assert result.constraints_C == []
        assert result.all_bindings == []
        assert result.num_interior_edges == 0
        assert result.sigma_min == DEFAULT_SIGMA_MIN

    def test_validation_report_default(self):
        """测试ValidationReport默认值"""
        report = ValidationReport()
        assert report.checks == {}
        assert report.failures == []
        assert report.curvature_violation == 0.0
        assert report.all_passed is True

    def test_trajectory_constraints_v1_default(self):
        """测试TrajectoryConstraints默认v1"""
        constraints = TrajectoryConstraints(
            max_velocity=2.0,
            max_acceleration=1.0,
            max_curvature=0.5,
        )
        assert constraints.curvature_constraint_version == "v1"
        assert constraints.sigma_min == "auto"

    def test_trajectory_constraints_v2(self):
        """测试TrajectoryConstraints v2模式"""
        constraints = TrajectoryConstraints(
            max_velocity=2.0,
            max_acceleration=1.0,
            max_curvature=0.5,
            curvature_constraint_version="v2",
            sigma_min=0.01,
        )
        assert constraints.curvature_constraint_version == "v2"
        assert constraints.sigma_min == 0.01

    def test_trajectory_constraints_invalid_version(self):
        """测试TrajectoryConstraints无效版本"""
        with pytest.raises(ValueError):
            TrajectoryConstraints(
                max_velocity=2.0,
                max_acceleration=1.0,
                max_curvature=0.5,
                curvature_constraint_version="v3",
            )

    def test_trajectory_constraints_negative_sigma_min(self):
        """测试TrajectoryConstraints负sigma_min"""
        with pytest.raises(ValueError):
            TrajectoryConstraints(
                max_velocity=2.0,
                max_acceleration=1.0,
                max_curvature=0.5,
                sigma_min=-0.01,
            )


# ============================================================
# 异常测试
# ============================================================
class TestExceptions:
    """异常层次测试"""

    def test_invalid_parameter_error(self):
        """测试InvalidParameterError"""
        with pytest.raises(InvalidParameterError):
            raise InvalidParameterError("test error")

    def test_solver_not_supported_error(self):
        """测试SolverNotSupportedError"""
        with pytest.raises(SolverNotSupportedError):
            raise SolverNotSupportedError("test error")

    def test_prerequisite_violation_error(self):
        """测试PrerequisiteViolationError"""
        with pytest.raises(PrerequisiteViolationError):
            raise PrerequisiteViolationError("test error")

    def test_constraint_construction_error(self):
        """测试ConstraintConstructionError"""
        with pytest.raises(ConstraintConstructionError):
            raise ConstraintConstructionError("test error")

    def test_vertex_extension_error(self):
        """测试VertexExtensionError"""
        with pytest.raises(VertexExtensionError):
            raise VertexExtensionError("test error")

    def test_exception_hierarchy(self):
        """测试异常继承关系"""
        from ackermann_gcs_pkg.curvature_constraint_v2 import CurvatureV2Error
        assert issubclass(InvalidParameterError, CurvatureV2Error)
        assert issubclass(SolverNotSupportedError, CurvatureV2Error)
        assert issubclass(PrerequisiteViolationError, CurvatureV2Error)
        assert issubclass(ConstraintConstructionError, CurvatureV2Error)
        assert issubclass(VertexExtensionError, CurvatureV2Error)


# ============================================================
# 常量测试
# ============================================================
class TestConstants:
    """数值安全常量测试"""

    def test_float_tol(self):
        assert CURVATURE_V2_FLOAT_TOL == 1e-12

    def test_singular_tol(self):
        assert CURVATURE_V2_SINGULAR_TOL == 1e-8

    def test_sigma_min_bounds(self):
        assert SIGMA_MIN_LOWER_BOUND == 1e-6
        assert SIGMA_MIN_UPPER_BOUND == 1.0
        assert SIGMA_MIN_LOWER_BOUND < SIGMA_MIN_UPPER_BOUND

    def test_default_sigma_min(self):
        assert DEFAULT_SIGMA_MIN == 0.01
        assert SIGMA_MIN_LOWER_BOUND < DEFAULT_SIGMA_MIN < SIGMA_MIN_UPPER_BOUND
