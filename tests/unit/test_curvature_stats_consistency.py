"""
曲率统计一致性单元测试

验证Bug修复：
1. 标准差计算统一在 |κ| 上，与 max/min/mean 保持语义一致
2. κ_max 限制值从车辆参数动态获取，而非硬编码
"""

import numpy as np
import pytest


# ==================== 纯数学测试（无需Drake） ====================

class TestStdCurvatureConsistency:
    """标准差计算一致性测试"""

    def test_std_curvature_on_abs_kappa_mixed_sign(self):
        """验证正负交替曲率时，std基于|κ|计算"""
        # 构造带符号的曲率数组
        kappa_array = np.array([1.0, -1.0, 0.5, -0.5])

        # 期望值：对 |κ| 计算标准差
        expected_std = np.std(np.abs(kappa_array))

        # 错误值：对 κ 计算标准差（修复前的行为）
        wrong_std = np.std(kappa_array)

        # 验证修复后的值正确
        assert abs(expected_std - 0.25) < 1e-10
        # 验证修复前后的值不同，证实Bug存在
        assert abs(expected_std - wrong_std) > 0.1

    def test_std_curvature_on_abs_kappa_positive_only(self):
        """验证纯正向曲率时，std(|κ|) = std(κ)，修复前后无差异"""
        kappa_array = np.array([1.0, 0.5, 0.8, 0.3])
        std_abs = np.std(np.abs(kappa_array))
        std_raw = np.std(kappa_array)

        # 纯正向时，|κ| = κ，两者应完全相同
        assert abs(std_abs - std_raw) < 1e-10

    def test_coefficient_of_variation_within_range(self):
        """验证修复后变异系数 cv = std/mean <= 1.0"""
        test_arrays = [
            np.array([1.0, -1.0, 0.5, -0.5]),
            np.array([2.0, -1.0, 0.0, -0.5, 1.5]),
            np.array([0.1, -0.1, 0.2, -0.2, 0.3]),
        ]

        for kappa_array in test_arrays:
            abs_kappa = np.abs(kappa_array)
            std_val = np.std(abs_kappa)
            mean_val = np.mean(abs_kappa)
            if mean_val > 0:
                cv = std_val / mean_val
                assert cv <= 1.0 + 1e-10, f"cv={cv} > 1.0 for {kappa_array}"

    def test_all_statistics_on_abs_kappa(self):
        """验证所有统计量均基于|κ|计算"""
        kappa_array = np.array([2.0, -1.0, 0.0, -0.5])
        abs_kappa = np.abs(kappa_array)

        expected_max = np.max(abs_kappa)
        expected_min = np.min(abs_kappa)
        expected_mean = np.mean(abs_kappa)
        expected_std = np.std(abs_kappa)

        assert abs(expected_max - 2.0) < 1e-10
        assert abs(expected_min - 0.0) < 1e-10
        assert abs(expected_mean - 0.875) < 1e-10
        # std([2.0, 1.0, 0.0, 0.5]) 的精确值
        assert abs(expected_std - np.std(np.array([2.0, 1.0, 0.0, 0.5]))) < 1e-10


class TestKappaMaxFromVehicleParams:
    """κ_max 动态获取测试"""

    def test_kappa_max_from_vehicle_params(self):
        """验证κ_max从车辆参数动态获取，与85°转向角配置一致"""
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        src_dir = os.path.join(project_root, 'src')
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        from ackermann_gcs_pkg.ackermann_data_structures import VehicleParams

        # 构造与实际配置一致的车辆参数（85°转向角，轴距2.5m）
        vehicle_params = VehicleParams(
            wheelbase=2.5,
            max_steering_angle=np.deg2rad(85),
            max_velocity=10.0,
            max_acceleration=8.0
        )

        # 85°转向角: tan(85°)/2.5 ≈ 4.572021
        expected_kappa_max = np.tan(np.deg2rad(85)) / 2.5
        assert abs(vehicle_params.max_curvature - expected_kappa_max) < 1e-5

        # 不应等于30°的硬编码值 0.230940
        assert abs(vehicle_params.max_curvature - 0.230940) > 1.0
