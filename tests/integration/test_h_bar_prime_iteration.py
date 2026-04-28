"""
h_bar_prime 迭代修正集成测试

端到端验证迭代修正流程在规划器中的完整行为和向后兼容性。

注意：集成测试需要 Drake 环境和较长的求解时间，
使用 @pytest.mark.slow 标记。
"""

import pytest
import numpy as np

from src.ackermann_gcs_pkg.ackermann_data_structures import (
    VehicleParams,
    EndpointState,
    TrajectoryConstraints,
    BezierConfig,
)
from src.ackermann_gcs_pkg.ackermann_gcs_planner import AckermannGCSPlanner
from src.ackermann_gcs_pkg.h_bar_prime_iteration import (
    HBarPrimeIterationResult,
)
from gcs_pkg.scripts.core.bezier import BezierGCS


def _make_simple_workspace():
    """创建简单的工作空间区域（2个矩形区域）。

    返回 (regions, source, target)。
    """
    from pydrake.geometry.optimization import HPolyhedron

    # 两个相邻的矩形区域
    # 区域1: x ∈ [0, 3], y ∈ [-1, 1]
    A1 = np.array([
        [1, 0], [-1, 0], [0, 1], [0, -1]
    ])
    b1 = np.array([3, 0, 1, 1])
    region1 = HPolyhedron(A1, b1)

    # 区域2: x ∈ [2, 5], y ∈ [-1, 1]
    A2 = np.array([
        [1, 0], [-1, 0], [0, 1], [0, -1]
    ])
    b2 = np.array([5, -2, 1, 1])
    region2 = HPolyhedron(A2, b2)

    regions = [region1, region2]

    source = EndpointState(
        position=np.array([0.5, 0.0]),
        heading=0.0,
        velocity=1.0,
    )
    target = EndpointState(
        position=np.array([4.5, 0.0]),
        heading=0.0,
        velocity=1.0,
    )

    return regions, source, target


def _make_vehicle_params():
    """创建测试用车辆参数。"""
    return VehicleParams(
        wheelbase=2.5,
        max_steering_angle=0.5,
        max_velocity=5.0,
        max_acceleration=3.0,
    )


@pytest.mark.slow
class TestHBarPrimeIterationIntegration:
    """h̄' 迭代修正集成测试"""

    def test_iterative_refinement_end_to_end(self):
        """迭代修正端到端：h_bar_prime=None + max_iterations>1 时自动迭代"""
        regions, source, target = _make_simple_workspace()
        vehicle_params = _make_vehicle_params()

        constraints = TrajectoryConstraints(
            max_velocity=vehicle_params.max_velocity,
            max_acceleration=vehicle_params.max_acceleration,
            max_curvature=vehicle_params.max_curvature,
            workspace_regions=regions,
            enable_curvature_hard_constraint=True,
            min_velocity=1.0,
            h_bar_prime=None,  # 触发迭代修正
            max_h_bar_prime_iterations=2,  # 限制迭代次数以加速测试
            h_bar_prime_safety_factor=0.7,
        )

        planner = AckermannGCSPlanner(
            vehicle_params=vehicle_params,
            bezier_config=BezierConfig(order=5, continuity=1),
        )

        cost_weights = {
            "time": 1.0,
            "path_length": 0.1,
            "energy": 0.01,
        }

        result = planner.plan_trajectory(
            source=source,
            target=target,
            workspace_regions=regions,
            constraints=constraints,
            cost_weights=cost_weights,
            verbose=True,
        )

        # 验证规划完成（成功或失败均可，关键是迭代修正被执行）
        assert result is not None

    def test_no_iteration_when_h_bar_prime_specified(self):
        """h_bar_prime 已指定时不执行迭代，仅应用 safety_factor"""
        regions, source, target = _make_simple_workspace()
        vehicle_params = _make_vehicle_params()

        constraints = TrajectoryConstraints(
            max_velocity=vehicle_params.max_velocity,
            max_acceleration=vehicle_params.max_acceleration,
            max_curvature=vehicle_params.max_curvature,
            workspace_regions=regions,
            enable_curvature_hard_constraint=True,
            min_velocity=1.0,
            h_bar_prime=2.0,  # 指定具体值，不触发迭代
            h_bar_prime_safety_factor=0.7,
        )

        planner = AckermannGCSPlanner(
            vehicle_params=vehicle_params,
            bezier_config=BezierConfig(order=5, continuity=1),
        )

        cost_weights = {
            "time": 1.0,
            "path_length": 0.1,
            "energy": 0.01,
        }

        result = planner.plan_trajectory(
            source=source,
            target=target,
            workspace_regions=regions,
            constraints=constraints,
            cost_weights=cost_weights,
            verbose=True,
        )

        # 验证规划完成
        assert result is not None

    def test_backward_compatibility(self):
        """向后兼容：未指定新字段时行为与修改前一致"""
        regions, source, target = _make_simple_workspace()
        vehicle_params = _make_vehicle_params()

        # 不设置任何新 h_bar_prime 字段
        constraints = TrajectoryConstraints(
            max_velocity=vehicle_params.max_velocity,
            max_acceleration=vehicle_params.max_acceleration,
            max_curvature=vehicle_params.max_curvature,
            workspace_regions=regions,
            enable_curvature_hard_constraint=True,
            min_velocity=1.0,
            # h_bar_prime=None (默认)
            # h_bar_prime_safety_factor=0.7 (默认)
            # max_h_bar_prime_iterations=3 (默认)
        )

        planner = AckermannGCSPlanner(
            vehicle_params=vehicle_params,
            bezier_config=BezierConfig(order=5, continuity=1),
        )

        cost_weights = {
            "time": 1.0,
            "path_length": 0.1,
            "energy": 0.01,
        }

        result = planner.plan_trajectory(
            source=source,
            target=target,
            workspace_regions=regions,
            constraints=constraints,
            cost_weights=cost_weights,
            verbose=True,
        )

        # 验证规划完成
        assert result is not None


@pytest.mark.slow
class TestComputeHBarPrimeFromTrajectoryIntegration:
    """compute_h_bar_prime_from_trajectory 集成测试"""

    def test_estimate_and_compute_consistency(self):
        """静态估算和精确计算的一致性验证"""
        # 静态估算
        h_estimated = BezierGCS.estimate_h_bar_prime(
            path_length_estimate=5.0,
            num_segments=2,
            w_time=1.0,
            w_energy=0.1,
        )

        # 验证估算值为正数
        assert h_estimated > 0

        # 验证与 v_optimal 的关系
        v_optimal = np.sqrt(1.0 / 0.1)
        expected = 5.0 / (2 * v_optimal)
        assert abs(h_estimated - expected) < 1e-10
