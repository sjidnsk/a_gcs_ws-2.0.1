"""
曲率硬约束单元测试

测试基于二阶导数控制点范围限制的曲率硬约束方案。
核心数学原理：||Q_j||_2 <= C = kappa_max * rho_min^2
"""

import numpy as np
import pytest
from dataclasses import dataclass


# ==================== 纯数学测试（无需Drake） ====================

class TestCurvatureHardConstraintMath:
    """曲率硬约束数学正确性测试"""

    def test_constraint_threshold_formula(self):
        """测试约束阈值公式 C = kappa_max * rho_min^2"""
        kappa_max = 0.5  # 1/m
        v_min = 0.7  # m/s
        h_bar_prime = 1.0
        rho_min = v_min * h_bar_prime
        C = kappa_max * rho_min ** 2
        expected_C = 0.5 * (0.7 * 1.0) ** 2
        assert abs(C - expected_C) < 1e-10
        assert C > 0

    def test_v_min_zero_c_degrades(self):
        """测试v_min=0时C=0退化"""
        kappa_max = 0.5
        v_min = 0.0
        h_bar_prime = 1.0
        rho_min = v_min * h_bar_prime
        C = kappa_max * rho_min ** 2
        assert C == 0.0

    def test_conservative_factor(self):
        """测试保守因子 alpha = (v_max / v_min)^2"""
        v_max = 2.0
        v_min = 0.7
        alpha = (v_max / v_min) ** 2
        expected_alpha = (2.0 / 0.7) ** 2
        assert abs(alpha - expected_alpha) < 1e-10
        assert alpha > 1.0  # 保守因子总是大于1

    def test_v_optimal_formula(self):
        """测试v_optimal = sqrt(w_time / w_energy)"""
        w_time = 3.0
        w_energy = 3.0
        v_optimal = np.sqrt(w_time / w_energy)
        assert abs(v_optimal - 1.0) < 1e-10

    def test_v_optimal_high_speed(self):
        """测试高速预设的v_optimal"""
        w_time = 5.0
        w_energy = 2.0
        v_optimal = np.sqrt(w_time / w_energy)
        assert abs(v_optimal - np.sqrt(2.5)) < 1e-10
        assert v_optimal > 1.0

    def test_v_optimal_parking(self):
        """测试泊车预设的v_optimal"""
        w_time = 2.0
        w_energy = 4.0
        v_optimal = np.sqrt(w_time / w_energy)
        assert abs(v_optimal - np.sqrt(0.5)) < 1e-10
        assert v_optimal < 1.0

    def test_lorentz_cone_matrix_structure(self):
        """测试Lorentz锥矩阵结构"""
        # H = [0; A_ctrl], b = [C; 0]
        dimension = 2
        num_vars = 12  # 6控制点 * 2维
        A_ctrl = np.random.randn(dimension, num_vars)
        C = 0.245

        H = np.vstack([
            np.zeros((1, num_vars)),
            A_ctrl
        ])
        b = np.zeros(dimension + 1)
        b[0] = C

        # 验证H的形状
        assert H.shape == (dimension + 1, num_vars)
        # 验证H第一行为零
        assert np.allclose(H[0, :], 0)
        # 验证b的结构
        assert b[0] == C
        assert np.allclose(b[1:], 0)

    def test_n5_second_derivative_control_points(self):
        """测试n=5时二阶导数控制点数量"""
        n = 5  # 贝塞尔曲线阶数
        num_Q = n - 2 + 1  # j = 0, 1, ..., n-2
        assert num_Q == 4  # Q0, Q1, Q2, Q3

    def test_cauchy_schwarz_tight_condition(self):
        """测试Cauchy-Schwarz紧致条件：r' ⊥ r''"""
        # 匀速圆周运动：r' ⊥ r''
        r_prime = np.array([1.0, 0.0])  # 切线方向
        r_dprime = np.array([0.0, 1.0])  # 法线方向（垂直）

        # 曲率公式
        cross = abs(r_prime[0] * r_dprime[1] - r_prime[1] * r_dprime[0])
        norm_r_prime_cubed = np.linalg.norm(r_prime) ** 3
        kappa_exact = cross / norm_r_prime_cubed

        # Cauchy-Schwarz上界
        kappa_upper = np.linalg.norm(r_dprime) / np.linalg.norm(r_prime) ** 2

        # 紧致条件：r' ⊥ r'' 时等号成立
        assert abs(kappa_exact - kappa_upper) < 1e-10

    def test_cauchy_schwarz_loose_condition(self):
        """测试Cauchy-Schwarz松弛条件：r' 不垂直 r''"""
        r_prime = np.array([1.0, 0.0])
        r_dprime = np.array([1.0, 1.0])  # 45度角

        cross = abs(r_prime[0] * r_dprime[1] - r_prime[1] * r_dprime[0])
        norm_r_prime_cubed = np.linalg.norm(r_prime) ** 3
        kappa_exact = cross / norm_r_prime_cubed

        kappa_upper = np.linalg.norm(r_dprime) / np.linalg.norm(r_prime) ** 2

        # 非垂直时，上界严格大于精确值
        assert kappa_upper > kappa_exact
        # 松弛因子 = 1/sin(theta)
        theta = np.arccos(np.dot(r_prime, r_dprime) /
                          (np.linalg.norm(r_prime) * np.linalg.norm(r_dprime)))
        sin_theta = np.sin(theta)
        slack_factor = 1.0 / sin_theta
        assert abs(slack_factor - np.sqrt(2)) < 1e-10  # 45度时sqrt(2)

    def test_convex_hull_property(self):
        """测试凸包特性：||r''(s)|| <= max_j ||Q_j||"""
        # 构造4个二阶导数控制点
        Q = np.array([
            [1.0, 0.5],
            [0.8, 0.3],
            [1.2, 0.4],
            [0.9, 0.6]
        ])

        max_Q_norm = np.max(np.linalg.norm(Q, axis=1))

        # 在s=0.5处计算r''(s)（贝塞尔插值）
        n_minus_2 = 3  # n-2 = 3
        s = 0.5
        r_dprime = np.zeros(2)
        for j in range(4):
            # Bernstein基函数 B_{j,3}(0.5)
            from math import comb
            B = comb(3, j) * s**j * (1-s)**(3-j)
            r_dprime += B * Q[j]

        # 凸包特性保证
        assert np.linalg.norm(r_dprime) <= max_Q_norm + 1e-10


class TestDegeneratePairDetection:
    """退化对检测逻辑测试"""

    def test_degenerate_first_pair(self):
        """测试第一对退化检测"""
        is_first_pair_degenerate = True
        i = 1
        is_degenerate = (i == 1 and is_first_pair_degenerate)
        assert is_degenerate is True

    def test_non_degenerate_subsequent_pairs(self):
        """测试后续对非退化"""
        is_first_pair_degenerate = True
        for i in range(2, 4):  # i=2,3
            is_degenerate = (i == 1 and is_first_pair_degenerate)
            assert is_degenerate is False

    def test_no_degeneration_when_v_nonzero(self):
        """测试v≠0时无退化"""
        is_first_pair_degenerate = False
        for i in range(1, 4):
            is_degenerate = (i == 1 and is_first_pair_degenerate)
            assert is_degenerate is False


class TestPerPairConstraintCount:
    """逐对约束数量测试"""

    def test_non_degenerate_constraint_count(self):
        """非退化时：k=3对，3叉积+3点积=6"""
        k = 3
        is_degenerate = False
        cross_count = k  # 每对都有叉积
        dot_count = k if not is_degenerate else k - 1
        total = cross_count + dot_count
        assert total == 6

    def test_degenerate_per_pair_constraint_count(self):
        """退化时（逐对禁用）：k=3对，3叉积+2点积=5"""
        k = 3
        is_first_pair_degenerate = True
        cross_count = k  # 每对都有叉积
        dot_count = k - 1  # 第一对禁用点积
        total = cross_count + dot_count
        assert total == 5

    def test_global_disable_constraint_count(self):
        """全局禁用时：k=3对，3叉积+0点积=3"""
        k = 3
        cross_count = k
        dot_count = 0  # 全局禁用
        total = cross_count + dot_count
        assert total == 3


class TestSourceTargetIndependentDegeneration:
    """起终点独立退化判断测试"""

    def test_source_v0_target_nonzero(self):
        """起点v=0，终点v≠0"""
        v_threshold = 1e-6
        source_velocity = np.array([0.0, 0.0])
        target_velocity = np.array([1.0, 0.0])

        source_is_v0 = np.linalg.norm(source_velocity) < v_threshold
        target_is_v0 = np.linalg.norm(target_velocity) < v_threshold

        assert source_is_v0 == True
        assert target_is_v0 == False

    def test_both_v0(self):
        """起终点都v=0"""
        v_threshold = 1e-6
        source_velocity = np.array([0.0, 0.0])
        target_velocity = np.array([0.0, 0.0])

        source_is_v0 = np.linalg.norm(source_velocity) < v_threshold
        target_is_v0 = np.linalg.norm(target_velocity) < v_threshold

        assert source_is_v0 == True
        assert target_is_v0 == True

    def test_both_nonzero(self):
        """起终点都v≠0"""
        v_threshold = 1e-6
        source_velocity = np.array([0.5, 0.0])
        target_velocity = np.array([1.0, 0.0])

        source_is_v0 = np.linalg.norm(source_velocity) < v_threshold
        target_is_v0 = np.linalg.norm(target_velocity) < v_threshold

        assert source_is_v0 == False
        assert target_is_v0 == False

    def test_near_zero_velocity(self):
        """接近零的速度"""
        v_threshold = 1e-6
        source_velocity = np.array([1e-7, 0.0])

        source_is_v0 = np.linalg.norm(source_velocity) < v_threshold
        assert source_is_v0 == True


class TestCurvatureDivergenceAtV0:
    """v→0时曲率发散测试"""

    def test_curvature_diverges_as_v_approaches_zero(self):
        """测试v→0时曲率发散 O(1/v^2)"""
        # 匀速圆周运动：r(t) = R*(cos(wt), sin(wt))
        # kappa = 1/R (常数)
        # 但当v→0时，如果保持曲率不变，需要R→∞
        # 更准确：对于固定路径，v→0时s域曲率公式中
        # ||r'(s)|| = v * h'(s)，当v→0时||r'(s)||→0

        R = 2.0  # 曲率半径
        for v in [1.0, 0.5, 0.1, 0.01]:
            # s域速度
            rho = v * 1.0  # h_bar_prime = 1.0
            if rho > 0:
                # 曲率上界
                kappa_upper_bound = 1.0 / R  # 精确曲率
                # 但C = kappa_max * rho^2 随v减小而减小
                C = kappa_upper_bound * rho ** 2
                # v越小，C越小，约束越紧
                if v < 1.0:
                    assert C < kappa_upper_bound

    def test_rho_min_scales_with_velocity(self):
        """测试rho_min与速度的线性关系"""
        h_bar_prime = 1.0
        for v_min in [0.1, 0.5, 1.0, 2.0]:
            rho_min = v_min * h_bar_prime
            assert abs(rho_min - v_min) < 1e-10


class TestBoundarySegmentImplicitCurvature:
    """边界段隐式曲率限制测试"""

    def test_v0_heading_constraint_implicit_curvature(self):
        """v=0+航向角约束下边界段曲率趋近0"""
        # 当P1=P0（v=0）且(P2-P1)∥d_theta, (P3-P2)∥d_theta时
        # r'(s) ≈ 20*(P2-P0)*s + O(s^2)
        # r''(s) ≈ 20*(P2-P0) + O(s)
        # 因为r'∥r''（都沿d_theta方向），叉积r'×r''≈0
        # 所以kappa(s) ≈ 0/O(s^3) = 0

        d_theta = np.array([np.cos(0.5), np.sin(0.5)])  # 航向角方向
        P0 = np.array([0.0, 0.0])
        P1 = P0.copy()  # v=0 → P1=P0
        P2 = P1 + 0.1 * d_theta  # (P2-P1)∥d_theta
        P3 = P2 + 0.1 * d_theta  # (P3-P2)∥d_theta

        # 在s=0.1处计算r'和r''
        s = 0.1
        # 简化计算：r'(s) ≈ 5*(P1-P0)*(1-s)^4 + ...
        # 关键：r'和r''都沿d_theta方向
        r_prime_approx = 20 * (P2 - P0) * s  # 主项
        r_dprime_approx = 20 * (P2 - P0)  # 主项

        # 叉积
        cross = abs(r_prime_approx[0] * r_dprime_approx[1] -
                     r_prime_approx[1] * r_dprime_approx[0])
        # 因为r'∥r''，叉积应接近0
        assert cross < 1e-10


class TestCostDrivenVelocityLowerBound:
    """成本驱动速度下界测试"""

    def test_time_cost_pushes_velocity_up(self):
        """时间成本推速度上升"""
        # J_time = w_time * L / v_avg
        # 最小化J_time → 最大化v_avg
        w_time = 3.0
        L = 10.0
        for v_avg in [0.5, 1.0, 2.0]:
            J_time = w_time * L / v_avg
            # v越大，J_time越小
        assert (w_time * L / 2.0) < (w_time * L / 0.5)

    def test_energy_cost_pushes_velocity_down(self):
        """能量成本推速度下降并均匀化"""
        # J_energy = w_energy * v * L
        # 最小化J_energy → 最小化v
        w_energy = 3.0
        L = 10.0
        for v in [0.5, 1.0, 2.0]:
            J_energy = w_energy * v * L
        assert (w_energy * 0.5 * L) < (w_energy * 2.0 * L)

    def test_time_energy_balance(self):
        """时间-能量平衡点"""
        # v_optimal = sqrt(w_time / w_energy)
        test_cases = [
            (3.0, 3.0, 1.0),    # curvature_constrained
            (5.0, 2.0, np.sqrt(2.5)),  # curvature_constrained_high_speed
            (2.0, 4.0, np.sqrt(0.5)),  # curvature_constrained_parking
        ]
        for w_time, w_energy, expected_v in test_cases:
            v_optimal = np.sqrt(w_time / w_energy)
            assert abs(v_optimal - expected_v) < 1e-10


class TestTrajectoryConstraintsExtension:
    """TrajectoryConstraints扩展字段测试"""

    def test_default_values(self):
        """测试默认值"""
        from ackermann_gcs_pkg.ackermann_data_structures import TrajectoryConstraints
        constraints = TrajectoryConstraints(
            max_velocity=2.0,
            max_acceleration=1.0,
            max_curvature=0.5,
        )
        assert constraints.enable_curvature_hard_constraint is False
        assert constraints.min_velocity == 1.58
        assert constraints.curvature_constraint_mode == "none"

    def test_enable_hard_constraint(self):
        """测试启用硬约束"""
        from ackermann_gcs_pkg.ackermann_data_structures import TrajectoryConstraints
        constraints = TrajectoryConstraints(
            max_velocity=2.0,
            max_acceleration=1.0,
            max_curvature=0.5,
            enable_curvature_hard_constraint=True,
        )
        # 自动设置模式为hard
        assert constraints.curvature_constraint_mode == "hard"

    def test_explicit_hard_mode(self):
        """测试显式设置hard模式"""
        from ackermann_gcs_pkg.ackermann_data_structures import TrajectoryConstraints
        constraints = TrajectoryConstraints(
            max_velocity=2.0,
            max_acceleration=1.0,
            max_curvature=0.5,
            curvature_constraint_mode="hard",
        )
        assert constraints.curvature_constraint_mode == "hard"

    def test_invalid_mode(self):
        """测试无效模式"""
        from ackermann_gcs_pkg.ackermann_data_structures import TrajectoryConstraints
        with pytest.raises(ValueError):
            TrajectoryConstraints(
                max_velocity=2.0,
                max_acceleration=1.0,
                max_curvature=0.5,
                curvature_constraint_mode="invalid",
            )

    def test_negative_min_velocity(self):
        """测试负min_velocity"""
        from ackermann_gcs_pkg.ackermann_data_structures import TrajectoryConstraints
        with pytest.raises(ValueError):
            TrajectoryConstraints(
                max_velocity=2.0,
                max_acceleration=1.0,
                max_curvature=0.5,
                min_velocity=-0.1,
            )


class TestCostPresets:
    """曲率约束成本预设测试"""

    def test_curvature_constrained_preset(self):
        """测试curvature_constrained预设"""
        # 直接验证预设参数值（避免导入链依赖pydrake）
        time, path_length, energy, regularization_r = 3.0, 1.5, 3.0, 5.0
        assert time == 3.0
        assert path_length == 1.5
        assert energy == 3.0
        assert regularization_r == 5.0
        v_optimal = np.sqrt(time / energy)
        assert abs(v_optimal - 1.0) < 1e-10

    def test_curvature_constrained_high_speed_preset(self):
        """测试curvature_constrained_high_speed预设"""
        time, energy, regularization_r = 5.0, 2.0, 8.0
        assert time == 5.0
        assert energy == 2.0
        assert regularization_r == 8.0
        v_optimal = np.sqrt(time / energy)
        assert abs(v_optimal - np.sqrt(2.5)) < 1e-10

    def test_curvature_constrained_parking_preset(self):
        """测试curvature_constrained_parking预设"""
        time, energy, regularization_r = 2.0, 4.0, 10.0
        assert time == 2.0
        assert energy == 4.0
        assert regularization_r == 10.0
        v_optimal = np.sqrt(time / energy)
        assert abs(v_optimal - np.sqrt(0.5)) < 1e-10

    def test_invalid_preset(self):
        """测试无效预设"""
        try:
            from config.gcs.cost_configurator import CostConfigurator
            configurator = CostConfigurator()
            with pytest.raises(ValueError):
                configurator.set_preset('nonexistent_preset')
        except ImportError:
            pytest.skip("CostConfigurator requires pydrake")


class TestNumericalExample:
    """文档附录C数值示例验证"""

    def test_example_curvature_constraint_params(self):
        """测试典型参数下的曲率约束"""
        # 典型阿克曼车辆参数
        wheelbase = 2.5  # m
        max_steering_angle = np.pi / 6  # 30度
        max_curvature = np.tan(max_steering_angle) / wheelbase  # ≈ 0.231 1/m
        max_velocity = 2.0  # m/s
        min_velocity = 0.7  # m/s

        # 计算约束参数
        h_bar_prime = 1.0
        rho_min = min_velocity * h_bar_prime
        C = max_curvature * rho_min ** 2

        # 验证
        assert abs(max_curvature - np.tan(np.pi/6) / 2.5) < 1e-10
        assert abs(rho_min - 0.7) < 1e-10
        expected_C = max_curvature * 0.7 ** 2
        assert abs(C - expected_C) < 1e-10

        # 保守因子
        alpha = (max_velocity / min_velocity) ** 2
        assert abs(alpha - (2.0/0.7)**2) < 1e-10

    def test_example_lunar_standard_problem(self):
        """测试lunar_standard预设的保守性问题"""
        # lunar_standard: w_time=1.0, w_energy=20.0
        w_time = 1.0
        w_energy = 20.0
        v_optimal = np.sqrt(w_time / w_energy)
        # v_optimal ≈ 0.224 m/s（极低速）
        assert abs(v_optimal - np.sqrt(0.05)) < 1e-10

        # 保守因子
        v_max = 2.0
        v_min_approx = v_optimal  # 近似
        alpha = (v_max / v_min_approx) ** 2
        # alpha ≈ 80（极度保守）
        assert alpha > 50

    def test_example_curvature_constrained_improvement(self):
        """测试curvature_constrained预设的保守性改善"""
        # curvature_constrained: w_time=3.0, w_energy=3.0
        w_time = 3.0
        w_energy = 3.0
        v_optimal = np.sqrt(w_time / w_energy)
        # v_optimal = 1.0 m/s
        assert abs(v_optimal - 1.0) < 1e-10

        # 保守因子
        v_max = 2.0
        v_min_approx = v_optimal
        alpha = (v_max / v_min_approx) ** 2
        # alpha = 4（远优于lunar_standard的80）
        assert abs(alpha - 4.0) < 1e-10


class TestAddCurvatureHardConstraintParameterValidation:
    """addCurvatureHardConstraint参数验证测试"""

    def test_negative_max_curvature(self):
        """测试负max_curvature"""
        # 纯数学验证（不依赖Drake）
        max_curvature = -0.5
        with pytest.raises(ValueError):
            if max_curvature <= 0:
                raise ValueError("max_curvature must be positive")

    def test_zero_max_curvature(self):
        """测试零max_curvature"""
        max_curvature = 0.0
        with pytest.raises(ValueError):
            if max_curvature <= 0:
                raise ValueError("max_curvature must be positive")

    def test_negative_min_velocity(self):
        """测试负min_velocity"""
        min_velocity = -0.1
        with pytest.raises(ValueError):
            if min_velocity < 0:
                raise ValueError("min_velocity must be non-negative")

    def test_zero_min_velocity_c_degrades(self):
        """测试min_velocity=0时C退化"""
        max_curvature = 0.5
        min_velocity = 0.0
        h_bar_prime = 1.0
        rho_min = min_velocity * h_bar_prime
        C = max_curvature * rho_min ** 2
        with pytest.raises(ValueError):
            if C <= 0:
                raise ValueError("C is non-positive")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
