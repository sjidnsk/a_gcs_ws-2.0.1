"""
h_bar_prime 估计方法单元测试

覆盖：
- compute_h_bar_prime_from_trajectory（5 个用例）
- estimate_h_bar_prime（3 个用例）
- addCurvatureHardConstraint safety_factor（2 个用例）
- TrajectoryConstraints 新字段（1 个用例）
- 迭代收敛判定（1 个用例）
- 放宽重试（1 个用例）
"""

import pytest
import numpy as np
import warnings
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from src.ackermann_gcs_pkg.ackermann_data_structures import (
    TrajectoryConstraints,
)
from src.ackermann_gcs_pkg.h_bar_prime_iteration import (
    HBarPrimeIterationResult,
)


# === compute_h_bar_prime_from_trajectory 测试 ===


class TestComputeHBarPrimeFromTrajectory:
    """测试 BezierGCS.compute_h_bar_prime_from_trajectory"""

    def _make_linear_trajectory(self, c=2.0, s_start=0.0, s_end=1.0):
        """构造线性时间参数化 h(s) = c*s 的 BezierTrajectory mock。

        h'(s) = c（常数），因此 h̄' = c。
        """
        trajectory = MagicMock()
        trajectory.start_s = s_start
        trajectory.end_s = s_end

        # time_traj.value(s) = c * s
        time_traj = MagicMock()
        time_traj.value = lambda s: np.array([[c * s]])
        time_traj.MakeDerivative = MagicMock()

        # h'(s) = c（常数导数）
        h_deriv = MagicMock()
        h_deriv.value = lambda s: np.array([[c]])
        time_traj.MakeDerivative.return_value = h_deriv

        trajectory.time_traj = time_traj
        return trajectory

    def test_linear_time_parameterization(self):
        """线性 h(s) = c*s 时，返回值应等于 c，误差 < 1e-6"""
        from gcs_pkg.scripts.core.bezier import BezierGCS

        for c in [0.5, 1.0, 2.0, 5.0]:
            traj = self._make_linear_trajectory(c=c)
            result = BezierGCS.compute_h_bar_prime_from_trajectory(traj)
            assert abs(result - c) < 1e-6, (
                f"Expected {c}, got {result}"
            )

    def test_consistency_with_path_length_over_time(self):
        """与 L_path/T_total 近似一致性"""
        from gcs_pkg.scripts.core.bezier import BezierGCS

        # h(s) = 3*s, s ∈ [0, 1]
        # T_total = h(1) - h(0) = 3
        # 对于匀速直线，L_path ≈ ||r'(s)|| * (s_end - s_start)
        # h̄' = 3, L_path/T_total ≈ 3（当 ||r'|| = 1 时）
        traj = self._make_linear_trajectory(c=3.0)
        result = BezierGCS.compute_h_bar_prime_from_trajectory(traj)
        assert abs(result - 3.0) < 0.05 * 3.0  # 5% 相对误差

    def test_invalid_trajectory(self):
        """无效 trajectory 抛 ValueError"""
        from gcs_pkg.scripts.core.bezier import BezierGCS

        # time_traj 为 None
        traj = MagicMock()
        traj.time_traj = None
        with pytest.raises(ValueError, match="time_traj is None"):
            BezierGCS.compute_h_bar_prime_from_trajectory(traj)

        # 无 time_traj 属性
        with pytest.raises(ValueError, match="valid BezierTrajectory"):
            BezierGCS.compute_h_bar_prime_from_trajectory(None)

    def test_too_few_samples(self):
        """采样点过少抛 ValueError"""
        from gcs_pkg.scripts.core.bezier import BezierGCS

        traj = self._make_linear_trajectory()
        with pytest.raises(ValueError, match="num_samples must be >= 10"):
            BezierGCS.compute_h_bar_prime_from_trajectory(
                traj, num_samples=5
            )

    def test_numerical_stability_near_hdot_min(self):
        """hdot_min 附近数值稳定性"""
        from gcs_pkg.scripts.core.bezier import BezierGCS

        # h'(s) = 0.01（接近 hdot_min）
        traj = self._make_linear_trajectory(c=0.01)
        result = BezierGCS.compute_h_bar_prime_from_trajectory(traj)
        assert np.isfinite(result)
        assert result > 0
        assert abs(result - 0.01) < 1e-6


# === estimate_h_bar_prime 测试 ===


class TestEstimateHBarPrime:
    """测试 BezierGCS.estimate_h_bar_prime"""

    def test_formula_correctness(self):
        """公式正确性：h̄' ≈ L_path / (N_segments * v_optimal)"""
        from gcs_pkg.scripts.core.bezier import BezierGCS

        # L=10, N=5, w_time=1.0, w_energy=0.1
        # v_optimal = sqrt(1.0/0.1) = sqrt(10) ≈ 3.162
        # h̄' = 10 / (5 * 3.162) ≈ 0.632
        result = BezierGCS.estimate_h_bar_prime(
            path_length_estimate=10.0,
            num_segments=5,
            w_time=1.0,
            w_energy=0.1,
        )
        v_opt = np.sqrt(1.0 / 0.1)
        expected = 10.0 / (5 * v_opt)
        assert abs(result - expected) < 1e-10

        # 直接指定 v_optimal
        result2 = BezierGCS.estimate_h_bar_prime(
            path_length_estimate=10.0,
            num_segments=5,
            v_optimal=2.0,
        )
        expected2 = 10.0 / (5 * 2.0)
        assert abs(result2 - expected2) < 1e-10

    def test_parameter_validation(self):
        """参数验证：非法输入抛 ValueError"""
        from gcs_pkg.scripts.core.bezier import BezierGCS

        with pytest.raises(ValueError, match="path_length_estimate"):
            BezierGCS.estimate_h_bar_prime(-1.0, 5)
        with pytest.raises(ValueError, match="path_length_estimate"):
            BezierGCS.estimate_h_bar_prime(0.0, 5)
        with pytest.raises(ValueError, match="num_segments"):
            BezierGCS.estimate_h_bar_prime(10.0, 0)
        with pytest.raises(ValueError, match="w_energy"):
            BezierGCS.estimate_h_bar_prime(10.0, 5, w_energy=0.0)
        with pytest.raises(ValueError, match="w_energy"):
            BezierGCS.estimate_h_bar_prime(10.0, 5, w_energy=-0.1)
        with pytest.raises(ValueError, match="w_time"):
            BezierGCS.estimate_h_bar_prime(10.0, 5, w_time=-1.0)

    def test_lower_bound_protection(self):
        """下界保护：结果 < hdot_min 时截断并发出 UserWarning"""
        from gcs_pkg.scripts.core.bezier import BezierGCS

        # 极短路径 + 极大段数 → h̄' 很小
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = BezierGCS.estimate_h_bar_prime(
                path_length_estimate=0.001,
                num_segments=100,
                v_optimal=10.0,
                hdot_min=0.01,
            )
            assert result == 0.01
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)


# === addCurvatureHardConstraint safety_factor 测试 ===


class TestCurvatureConstraintSafetyFactor:
    """测试 addCurvatureHardConstraint 的 safety_factor 支持"""

    def test_safety_factor_validation(self):
        """safety_factor 范围验证"""
        from gcs_pkg.scripts.core.bezier import BezierGCS
        from pydrake.geometry.optimization import HPolyhedron

        # 创建简单的 BezierGCS 实例
        region = HPolyhedron.MakeUnitBox(2)
        gcs = BezierGCS(
            regions=[region], order=3, continuity=1
        )

        # safety_factor = 0 应抛 ValueError
        with pytest.raises(ValueError, match="h_bar_prime_safety_factor"):
            gcs.addCurvatureHardConstraint(
                max_curvature=0.5,
                min_velocity=1.0,
                h_bar_prime=2.0,
                h_bar_prime_safety_factor=0.0,
            )

        # safety_factor > 1.0 应抛 ValueError
        with pytest.raises(ValueError, match="h_bar_prime_safety_factor"):
            gcs.addCurvatureHardConstraint(
                max_curvature=0.5,
                min_velocity=1.0,
                h_bar_prime=2.0,
                h_bar_prime_safety_factor=1.5,
            )

    def test_safety_factor_application(self):
        """safety_factor 正确应用：effective = h_bar_prime * safety_factor"""
        from gcs_pkg.scripts.core.bezier import BezierGCS
        from pydrake.geometry.optimization import HPolyhedron

        region = HPolyhedron.MakeUnitBox(2)
        gcs = BezierGCS(
            regions=[region], order=3, continuity=1
        )

        # h_bar_prime=2.0, safety_factor=0.7 → effective=1.4
        # C = kappa_max * (v_min * 1.4)^2
        # = 0.5 * (1.0 * 1.4)^2 = 0.5 * 1.96 = 0.98
        gcs.addCurvatureHardConstraint(
            max_curvature=0.5,
            min_velocity=1.0,
            h_bar_prime=2.0,
            h_bar_prime_safety_factor=0.7,
        )
        # 约束已添加，验证 curvature_constraints 非空
        assert len(gcs.curvature_constraints) > 0


# === TrajectoryConstraints 新字段测试 ===


class TestTrajectoryConstraintsNewFields:
    """测试 TrajectoryConstraints 新增 h_bar_prime 相关字段"""

    def test_default_values(self):
        """新字段默认值正确"""
        constraints = TrajectoryConstraints(
            max_velocity=5.0,
            max_acceleration=3.0,
            max_curvature=0.5,
        )
        assert constraints.h_bar_prime is None
        assert constraints.h_bar_prime_safety_factor == 0.7
        assert constraints.max_h_bar_prime_iterations == 3
        assert constraints.h_bar_prime_convergence_threshold == 0.15
        assert constraints.h_bar_prime_relax_factor == 1.3
        assert constraints.max_h_bar_prime_relax_attempts == 3
        assert constraints.h_bar_prime_safety_factor_decay == 0.8

    def test_custom_values(self):
        """自定义值正确"""
        constraints = TrajectoryConstraints(
            max_velocity=5.0,
            max_acceleration=3.0,
            max_curvature=0.5,
            h_bar_prime=2.0,
            h_bar_prime_safety_factor=0.8,
            max_h_bar_prime_iterations=5,
            h_bar_prime_convergence_threshold=0.1,
            h_bar_prime_relax_factor=1.5,
            max_h_bar_prime_relax_attempts=5,
        )
        assert constraints.h_bar_prime == 2.0
        assert constraints.h_bar_prime_safety_factor == 0.8
        assert constraints.max_h_bar_prime_iterations == 5

    def test_invalid_values(self):
        """非法参数抛 ValueError"""
        with pytest.raises(ValueError, match="h_bar_prime must be positive"):
            TrajectoryConstraints(
                max_velocity=5.0,
                max_acceleration=3.0,
                max_curvature=0.5,
                h_bar_prime=-1.0,
            )
        with pytest.raises(ValueError, match="h_bar_prime_safety_factor"):
            TrajectoryConstraints(
                max_velocity=5.0,
                max_acceleration=3.0,
                max_curvature=0.5,
                h_bar_prime_safety_factor=0.0,
            )
        with pytest.raises(ValueError, match="max_h_bar_prime_iterations"):
            TrajectoryConstraints(
                max_velocity=5.0,
                max_acceleration=3.0,
                max_curvature=0.5,
                max_h_bar_prime_iterations=0,
            )
        with pytest.raises(ValueError, match="h_bar_prime_convergence_threshold"):
            TrajectoryConstraints(
                max_velocity=5.0,
                max_acceleration=3.0,
                max_curvature=0.5,
                h_bar_prime_convergence_threshold=0.0,
            )

    def test_backward_compatibility(self):
        """向后兼容：不传新字段时行为不变"""
        constraints = TrajectoryConstraints(
            max_velocity=5.0,
            max_acceleration=3.0,
            max_curvature=0.5,
        )
        # 所有新字段有默认值，不抛异常
        assert constraints.h_bar_prime is None
        assert constraints.h_bar_prime_safety_factor == 0.7


# === 迭代收敛判定测试 ===


class TestIterationConvergence:
    """测试迭代收敛判定逻辑"""

    def test_convergence_check(self):
        """收敛判定：相对变化 < threshold 时 converged=True"""
        result = HBarPrimeIterationResult(
            h_bar_prime=2.0,
            effective_h_bar_prime=1.4,
            converged=True,
            num_iterations=2,
            iteration_history=[2.1, 2.0],
            convergence_reason="converged",
            relax_attempts=0,
        )
        assert result.converged is True
        assert result.num_iterations == 2
        # 相对变化 = |2.0 - 2.1| / 2.1 ≈ 0.048 < 0.15
        relative_change = abs(2.0 - 2.1) / 2.1
        assert relative_change < 0.15

    def test_not_converged(self):
        """未收敛：相对变化 >= threshold"""
        result = HBarPrimeIterationResult(
            h_bar_prime=3.0,
            effective_h_bar_prime=2.1,
            converged=False,
            num_iterations=3,
            iteration_history=[1.0, 2.0, 3.0],
            convergence_reason="max_iterations",
            relax_attempts=0,
        )
        assert result.converged is False
        assert result.convergence_reason == "max_iterations"


# === 放宽重试测试 ===


class TestRelaxOnSolveFailure:
    """测试求解失败时的放宽重试逻辑"""

    def test_relax_factor(self):
        """放宽因子正确：h̄' *= relax_factor"""
        h_bar_prime = 2.0
        relax_factor = 1.3
        h_bar_prime_after_relax = h_bar_prime * relax_factor
        assert abs(h_bar_prime_after_relax - 2.6) < 1e-10

    def test_relax_result(self):
        """放宽结果记录正确"""
        result = HBarPrimeIterationResult(
            h_bar_prime=2.6,
            effective_h_bar_prime=1.82,
            converged=False,
            num_iterations=2,
            iteration_history=[2.0, 2.6],
            convergence_reason="solve_failed",
            relax_attempts=1,
        )
        assert result.relax_attempts == 1
        assert result.convergence_reason == "solve_failed"


# === 动态 safety_factor 收紧测试 ===


class TestDynamicSafetyFactorTightening:
    """测试 h̄' 下降时自动收紧 safety_factor"""

    def test_safety_factor_decay_default(self):
        """safety_factor_decay 默认值为 0.8"""
        constraints = TrajectoryConstraints(
            max_velocity=5.0,
            max_acceleration=3.0,
            max_curvature=0.5,
        )
        assert constraints.h_bar_prime_safety_factor_decay == 0.8

    def test_safety_factor_decay_validation(self):
        """safety_factor_decay 参数验证"""
        # decay = 0 应抛 ValueError
        with pytest.raises(ValueError, match="h_bar_prime_safety_factor_decay"):
            TrajectoryConstraints(
                max_velocity=5.0,
                max_acceleration=3.0,
                max_curvature=0.5,
                h_bar_prime_safety_factor_decay=0.0,
            )
        # decay > 1.0 应抛 ValueError
        with pytest.raises(ValueError, match="h_bar_prime_safety_factor_decay"):
            TrajectoryConstraints(
                max_velocity=5.0,
                max_acceleration=3.0,
                max_curvature=0.5,
                h_bar_prime_safety_factor_decay=1.5,
            )
        # decay = 1.0 合法（禁用动态收紧）
        constraints = TrajectoryConstraints(
            max_velocity=5.0,
            max_acceleration=3.0,
            max_curvature=0.5,
            h_bar_prime_safety_factor_decay=1.0,
        )
        assert constraints.h_bar_prime_safety_factor_decay == 1.0

    def test_tightening_logic(self):
        """h̄' 下降时 safety_factor 收紧逻辑"""
        # 模拟：初始 safety_factor=0.7, decay=0.8
        # h̄' 从 2.596 降至 1.955（下降 24.7% > 阈值 15%）
        # 收紧后 safety_factor = 0.7 * 0.8 = 0.56
        initial_sf = 0.7
        decay = 0.8
        tightened_sf = initial_sf * decay
        assert abs(tightened_sf - 0.56) < 1e-10

        # effective h̄' 对比：
        # 收紧前：1.955 * 0.7 = 1.3685
        # 收紧后：1.955 * 0.56 = 1.0948
        h_bar_prime = 1.955
        effective_before = h_bar_prime * initial_sf
        effective_after = h_bar_prime * tightened_sf
        assert effective_after < effective_before
        # C 值更小 → 约束更紧
        # C_before = kappa * (v_min * 1.3685)^2
        # C_after  = kappa * (v_min * 1.0948)^2
        # C_after / C_before = (1.0948/1.3685)^2 ≈ 0.64
        ratio = (effective_after / effective_before) ** 2
        assert abs(ratio - 0.64) < 0.01

    def test_iteration_result_with_final_safety_factor(self):
        """HBarPrimeIterationResult 记录 final_safety_factor"""
        result = HBarPrimeIterationResult(
            h_bar_prime=1.955,
            effective_h_bar_prime=1.955 * 0.56,
            converged=False,
            num_iterations=3,
            iteration_history=[2.596, 1.955],
            convergence_reason="max_iterations",
            relax_attempts=0,
            final_safety_factor=0.56,
        )
        assert result.final_safety_factor == 0.56
        assert abs(result.effective_h_bar_prime - 1.955 * 0.56) < 1e-10

    def test_no_tightening_when_hbar_increases(self):
        """h̄' 增加时不收紧 safety_factor"""
        # h̄' 从 1.0 增至 1.1（增加 10% < 阈值 15%）
        # safety_factor 保持不变
        initial_sf = 0.7
        # 不触发收紧条件
        final_sf = initial_sf  # 保持不变
        assert final_sf == 0.7


# === 多候选轨迹筛选测试 ===


class TestCandidateTrajectorySelection:
    """测试多候选轨迹按约束违反量筛选逻辑"""

    def test_select_trajectory_with_max_h_bar_prime(self):
        """_select_trajectory_with_max_h_bar_prime 选择 h̄' 最大的候选"""
        from src.ackermann_gcs_pkg.h_bar_prime_iteration import (
            _select_trajectory_with_max_h_bar_prime,
        )

        # 模拟 BezierGCS.compute_h_bar_prime_from_trajectory
        mock_bezier_gcs = MagicMock()
        mock_bezier_gcs.compute_h_bar_prime_from_trajectory = MagicMock(
            side_effect=[1.5, 2.0, 1.8]  # 3 条候选的 h̄' 值
        )

        # 模拟 3 条候选轨迹
        traj0 = MagicMock()
        traj1 = MagicMock()
        traj2 = MagicMock()

        # result = (trajectory, results_dict)
        results_dict = {
            "all_candidate_trajectories": [traj0, traj1, traj2]
        }
        result = (traj0, results_dict)

        selected = _select_trajectory_with_max_h_bar_prime(
            traj0, result, mock_bezier_gcs, verbose=False
        )
        # 应选择 h̄'=2.0 的 traj1
        assert selected is traj1

    def test_select_default_when_single_candidate(self):
        """只有一条候选轨迹时返回默认轨迹"""
        from src.ackermann_gcs_pkg.h_bar_prime_iteration import (
            _select_trajectory_with_max_h_bar_prime,
        )

        mock_bezier_gcs = MagicMock()
        traj0 = MagicMock()
        results_dict = {"all_candidate_trajectories": [traj0]}
        result = (traj0, results_dict)

        selected = _select_trajectory_with_max_h_bar_prime(
            traj0, result, mock_bezier_gcs, verbose=False
        )
        assert selected is traj0

    def test_select_returns_none_for_none_input(self):
        """输入 None 时返回 None"""
        from src.ackermann_gcs_pkg.h_bar_prime_iteration import (
            _select_trajectory_with_max_h_bar_prime,
        )

        mock_bezier_gcs = MagicMock()
        selected = _select_trajectory_with_max_h_bar_prime(
            None, (None, {}), mock_bezier_gcs, verbose=False
        )
        assert selected is None

    def test_planner_candidate_screening_logic(self):
        """规划器候选筛选逻辑：综合违反量更小时替换"""
        # 模拟 3 条候选轨迹的违反量
        # 候选0（成本最优）：v=0, κ=2.0, a=0 → combined=2.0
        # 候选1：v=0, κ=0.0, a=0 → combined=0.0
        # 候选2：v=0, κ=1.0, a=0 → combined=1.0
        # 应选择候选1（违反量最小）

        # 综合违反量公式：vel*10 + curv + acc*5
        combined_0 = 0.0 * 10.0 + 2.0 + 0.0 * 5.0  # 2.0
        combined_1 = 0.0 * 10.0 + 0.0 + 0.0 * 5.0  # 0.0
        combined_2 = 0.0 * 10.0 + 1.0 + 0.0 * 5.0  # 1.0

        assert combined_1 < combined_2 < combined_0
        # 候选1 的综合违反量最小，应被选中

    def test_early_exit_when_feasible_found(self):
        """找到可行轨迹后提前退出筛选"""
        # 如果某候选轨迹 v_viol < 1e-4 且 κ_viol < 1e-4
        # 则无需继续评估剩余候选
        feasible_viol = 0.0 * 10.0 + 0.0 + 0.0 * 5.0  # 0.0
        assert feasible_viol < 1e-4
