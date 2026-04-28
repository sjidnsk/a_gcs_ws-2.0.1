"""
曲率硬约束集成测试

测试曲率硬约束与GCS框架的集成，包括：
- Drake API兼容性（LorentzConeConstraint构建）
- 约束添加到GCS边
- v=0航向角协同
- 逐对选择性禁用
"""

import numpy as np
import pytest


# ==================== Drake API集成测试 ====================

try:
    from pydrake.solvers import LorentzConeConstraint, Binding, Constraint
    from pydrake.symbolic import (
        DecomposeLinearExpressions,
        MakeMatrixContinuousVariable,
        MakeVectorContinuousVariable,
    )
    from pydrake.geometry.optimization import HPolyhedron
    DRAKE_AVAILABLE = True
except ImportError:
    DRAKE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not DRAKE_AVAILABLE,
    reason="Drake not available"
)


class TestLorentzConeConstraintConstruction:
    """LorentzConeConstraint构建测试"""

    def test_curvature_constraint_matrix_construction(self):
        """测试曲率约束的Lorentz锥矩阵构建"""
        # 模拟二阶导数控制点的线性表达式分解
        dimension = 2
        num_vars = 12  # 6控制点 * 2维

        # 模拟A_ctrl矩阵（二阶导数系数）
        A_ctrl = np.random.randn(dimension, num_vars)
        C = 0.245  # kappa_max * rho_min^2

        # 构建H和b
        H = np.vstack([
            np.zeros((1, num_vars)),
            A_ctrl
        ])
        b = np.zeros(dimension + 1)
        b[0] = C

        # 创建LorentzConeConstraint
        constraint = LorentzConeConstraint(H, b)

        # 验证约束已创建
        assert constraint is not None

    def test_velocity_constraint_matrix_construction(self):
        """测试速度约束的Lorentz锥矩阵构建（对比参考）"""
        dimension = 2
        num_vars = 12

        A_ctrl = np.random.randn(dimension, num_vars)
        b_ctrl = np.random.randn(1, num_vars)
        max_velocity = 2.0

        # 速度约束：H = [v_max * b_ctrl; A_ctrl]
        H = np.vstack([
            max_velocity * b_ctrl,
            A_ctrl
        ])

        constraint = LorentzConeConstraint(H, np.zeros(H.shape[0]))
        assert constraint is not None

    def test_curvature_vs_velocity_constraint_structure(self):
        """测试曲率约束与速度约束的结构差异"""
        dimension = 2
        num_vars = 12
        A_ctrl = np.random.randn(dimension, num_vars)
        b_ctrl = np.random.randn(1, num_vars)

        # 速度约束
        H_vel = np.vstack([2.0 * b_ctrl, A_ctrl])
        b_vel = np.zeros(dimension + 1)

        # 曲率约束
        C = 0.245
        H_curv = np.vstack([np.zeros((1, num_vars)), A_ctrl])
        b_curv = np.zeros(dimension + 1)
        b_curv[0] = C

        # 关键差异：
        # 1. 速度约束H第一行 = v_max * b_ctrl（非常数）
        # 2. 曲率约束H第一行 = 0（常数），C在b中
        assert not np.allclose(H_vel[0, :], H_curv[0, :])
        assert np.allclose(H_curv[0, :], 0)
        assert not np.allclose(b_vel, b_curv)
        assert b_curv[0] == C


class TestDecomposeLinearExpressions:
    """DecomposeLinearExpressions集成测试"""

    def test_second_derivative_control_points(self):
        """测试二阶导数控制点的线性表达式分解"""
        from pydrake.symbolic import Expression
        from pydrake.trajectories import BsplineTrajectory_
        from pydrake.math import BsplineBasis_, KnotVectorType

        order = 5
        dimension = 2

        # 创建符号变量
        u_control = MakeMatrixContinuousVariable(dimension, order + 1, "xu")
        u_duration = MakeVectorContinuousVariable(order + 1, "Tu")
        u_vars = np.concatenate((u_control.flatten("F"), u_duration))

        # 创建符号轨迹
        u_r_trajectory = BsplineTrajectory_[Expression](
            BsplineBasis_[Expression](order + 1, order + 1,
                                       KnotVectorType.kClampedUniform, 0., 1.),
            u_control)

        # 获取二阶导数控制点
        u_path_ddot = u_r_trajectory.MakeDerivative(2).control_points()

        # 验证二阶导数控制点数量
        assert len(u_path_ddot) == order - 2 + 1  # n-2+1 = 4

        # 验证每个控制点可以分解为线性表达式
        for ii in range(len(u_path_ddot)):
            A_ctrl = DecomposeLinearExpressions(u_path_ddot[ii], u_vars)
            assert A_ctrl.shape[0] == dimension
            assert A_ctrl.shape[1] == len(u_vars)


class TestHeadingConstraintPerPairIntegration:
    """逐对选择性禁用航向角约束集成测试"""

    def test_per_pair_constraint_creation(self):
        """测试逐对约束创建"""
        from ackermann_gcs_pkg.rotation_matrix_heading_constraint import (
            HeadingConstraintConfig,
            HeadingConstraintMethod,
            HeadingConstraintFactory,
        )
        from pydrake.symbolic import Variable

        # 创建符号变量
        num_vars = 12
        variables = [Variable(f"x{i}") for i in range(num_vars)]

        # 创建控制点（符号表达式）
        control_points = []
        for i in range(4):
            control_points.append((variables[i*2], variables[i*2+1]))

        config = HeadingConstraintConfig(
            method=HeadingConstraintMethod.ROTATION_MATRIX,
            num_control_points=3,
            enable_multi_point=True,
            enable_direction_constraint=True,
            direction_epsilon=0.01,
        )

        # 非退化情况
        constraints_normal = HeadingConstraintFactory.create_heading_constraints_per_pair(
            heading_angle=0.5,
            control_points=control_points,
            variables=variables,
            config=config,
            constraint_type='source',
            is_first_pair_degenerate=False,
        )
        # k=3对：3叉积+3点积=6
        assert len(constraints_normal) == 6

        # 退化情况
        constraints_degenerate = HeadingConstraintFactory.create_heading_constraints_per_pair(
            heading_angle=0.5,
            control_points=control_points,
            variables=variables,
            config=config,
            constraint_type='source',
            is_first_pair_degenerate=True,
        )
        # k=3对：3叉积+2点积=5（第一对禁用点积）
        assert len(constraints_degenerate) == 5

    def test_per_pair_vs_standard_consistency(self):
        """测试逐对方法与标准方法在非退化时的一致性"""
        from ackermann_gcs_pkg.rotation_matrix_heading_constraint import (
            HeadingConstraintConfig,
            HeadingConstraintMethod,
            HeadingConstraintFactory,
        )
        from pydrake.symbolic import Variable

        num_vars = 12
        variables = [Variable(f"x{i}") for i in range(num_vars)]
        control_points = []
        for i in range(4):
            control_points.append((variables[i*2], variables[i*2+1]))

        config = HeadingConstraintConfig(
            method=HeadingConstraintMethod.ROTATION_MATRIX,
            num_control_points=3,
            enable_multi_point=True,
            enable_direction_constraint=True,
        )

        # 标准方法
        constraints_standard = HeadingConstraintFactory.create_heading_constraints(
            heading_angle=0.5,
            control_points=control_points,
            variables=variables,
            config=config,
            constraint_type='source',
        )

        # 逐对方法（非退化）
        constraints_per_pair = HeadingConstraintFactory.create_heading_constraints_per_pair(
            heading_angle=0.5,
            control_points=control_points,
            variables=variables,
            config=config,
            constraint_type='source',
            is_first_pair_degenerate=False,
        )

        # 非退化时，两种方法应产生相同数量的约束
        assert len(constraints_standard) == len(constraints_per_pair)


class TestCurvatureConstraintMode:
    """曲率约束模式测试"""

    def test_curvature_constraint_mode_enum(self):
        """测试曲率约束模式枚举"""
        from ackermann_gcs_pkg.ackermann_data_structures import CurvatureConstraintMode
        assert CurvatureConstraintMode.NONE.value == "none"
        assert CurvatureConstraintMode.HARD.value == "hard"
        assert CurvatureConstraintMode.TURNING_RADIUS.value == "turning_radius"

    def test_trajectory_constraints_with_hard_mode(self):
        """测试TrajectoryConstraints的hard模式"""
        from ackermann_gcs_pkg.ackermann_data_structures import TrajectoryConstraints
        constraints = TrajectoryConstraints(
            max_velocity=2.0,
            max_acceleration=1.0,
            max_curvature=0.5,
            enable_curvature_hard_constraint=True,
            min_velocity=0.7,
        )
        assert constraints.enable_curvature_hard_constraint is True
        assert constraints.curvature_constraint_mode == "hard"
        assert constraints.min_velocity == 0.7


class TestBezierGCSAddCurvatureHardConstraint:
    """BezierGCS.addCurvatureHardConstraint集成测试"""

    def test_add_curvature_hard_constraint_creates_constraints(self):
        """测试addCurvatureHardConstraint创建约束"""
        from gcs_pkg.scripts.core.bezier import BezierGCS

        # 创建简单的GCS
        region = HPolyhedron.MakeUnitBox(2)
        regions = [region, region]

        bezier_gcs = BezierGCS(
            regions=regions,
            order=5,
            continuity=1,
        )

        # 记录约束数量
        initial_constraint_count = len(bezier_gcs.deriv_constraints)

        # 添加曲率硬约束
        bezier_gcs.addCurvatureHardConstraint(
            max_curvature=0.5,
            min_velocity=0.7,
        )

        # 验证约束已添加
        # n=5时，二阶导数有4个控制点，每个添加1个Lorentz锥约束
        assert len(bezier_gcs.deriv_constraints) > initial_constraint_count

    def test_add_curvature_hard_constraint_zero_min_velocity_raises(self):
        """测试min_velocity=0时抛出异常"""
        from gcs_pkg.scripts.core.bezier import BezierGCS

        region = HPolyhedron.MakeUnitBox(2)
        regions = [region, region]

        bezier_gcs = BezierGCS(
            regions=regions,
            order=5,
            continuity=1,
        )

        with pytest.raises(ValueError):
            bezier_gcs.addCurvatureHardConstraint(
                max_curvature=0.5,
                min_velocity=0.0,
            )

    def test_add_curvature_hard_constraint_negative_curvature_raises(self):
        """测试负曲率抛出异常"""
        from gcs_pkg.scripts.core.bezier import BezierGCS

        region = HPolyhedron.MakeUnitBox(2)
        regions = [region, region]

        bezier_gcs = BezierGCS(
            regions=regions,
            order=5,
            continuity=1,
        )

        with pytest.raises(ValueError):
            bezier_gcs.addCurvatureHardConstraint(
                max_curvature=-0.5,
                min_velocity=0.7,
            )

    def test_add_curvature_hard_constraint_for_edges(self):
        """测试addCurvatureHardConstraintForEdges跳过边界边"""
        from gcs_pkg.scripts.core.bezier import BezierGCS

        region = HPolyhedron.MakeUnitBox(2)
        regions = [region, region]

        bezier_gcs = BezierGCS(
            regions=regions,
            order=5,
            continuity=1,
        )

        # 获取边界边id
        boundary_edge_ids = set()
        for edge in bezier_gcs.gcs.Edges():
            if edge.u() == bezier_gcs.source:
                # source边的下一条边是边界边
                for next_edge in bezier_gcs.gcs.Edges():
                    if next_edge.u() == edge.v():
                        boundary_edge_ids.add(id(next_edge))
                        break
            break

        # 添加曲率硬约束（跳过边界边）
        initial_count = len(bezier_gcs.deriv_constraints)
        bezier_gcs.addCurvatureHardConstraintForEdges(
            max_curvature=0.5,
            min_velocity=0.7,
            boundary_edge_ids=boundary_edge_ids,
        )
        assert len(bezier_gcs.deriv_constraints) > initial_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
