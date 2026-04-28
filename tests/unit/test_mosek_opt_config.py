"""
MosekOptimizationConfig 单元测试

覆盖需求: REQ-001, REQ-003, REQ-005, REQ-010, NFR-006
测试用例: UT-001 ~ UT-005
"""

import pytest
import os

from config.solver.mosek_opt_config import MosekOptimizationConfig


class TestMosekOptimizationConfig:
    """MosekOptimizationConfig 单元测试"""

    # --- UT-001: 默认值正确性 ---
    def test_default_values(self):
        """验证所有默认值正确"""
        config = MosekOptimizationConfig()
        assert config.max_paths == 5
        assert config.num_threads == 8
        assert config.mio_max_time == 30.0
        assert config.max_rounding_attempts == 3
        assert config.max_rounded_paths == 5
        assert config.enable_reduced_paths is True
        assert config.enable_thread_limit is True
        assert config.enable_mio_time_limit is True
        assert config.enable_incremental_phi is True

    # --- UT-002: effective_max_paths 开关逻辑 ---
    def test_effective_max_paths_enabled(self):
        """enable_reduced_paths=True时返回max_paths"""
        config = MosekOptimizationConfig(max_paths=3, enable_reduced_paths=True)
        assert config.effective_max_paths() == 3

    def test_effective_max_paths_disabled(self):
        """enable_reduced_paths=False时返回原始值10"""
        config = MosekOptimizationConfig(max_paths=3, enable_reduced_paths=False)
        assert config.effective_max_paths() == 10

    # --- UT-003: effective_num_threads 开关逻辑 ---
    def test_effective_num_threads_enabled(self):
        """enable_thread_limit=True时返回num_threads"""
        config = MosekOptimizationConfig(num_threads=2, enable_thread_limit=True)
        assert config.effective_num_threads() == 2

    def test_effective_num_threads_disabled(self):
        """enable_thread_limit=False时返回原始值8"""
        config = MosekOptimizationConfig(num_threads=2, enable_thread_limit=False)
        assert config.effective_num_threads() == 8

    # --- UT-004: effective_mio_max_time 开关逻辑 ---
    def test_effective_mio_max_time_enabled(self):
        """enable_mio_time_limit=True时返回mio_max_time"""
        config = MosekOptimizationConfig(mio_max_time=30.0, enable_mio_time_limit=True)
        assert config.effective_mio_max_time() == 30.0

    def test_effective_mio_max_time_disabled(self):
        """enable_mio_time_limit=False时返回原始值3600.0"""
        config = MosekOptimizationConfig(mio_max_time=30.0, enable_mio_time_limit=False)
        assert config.effective_mio_max_time() == 3600.0

    # --- UT-005: 参数范围校验 ---
    def test_max_paths_too_low(self):
        """max_paths < 1 抛出ValueError"""
        with pytest.raises(ValueError, match="max_paths"):
            MosekOptimizationConfig(max_paths=0)

    def test_max_paths_too_high(self):
        """max_paths > 20 抛出ValueError"""
        with pytest.raises(ValueError, match="max_paths"):
            MosekOptimizationConfig(max_paths=21)

    def test_num_threads_too_low(self):
        """num_threads < 1 抛出ValueError"""
        with pytest.raises(ValueError, match="num_threads"):
            MosekOptimizationConfig(num_threads=0)

    def test_num_threads_too_high(self):
        """num_threads > CPU核心数 抛出ValueError"""
        cpu_count = os.cpu_count() or 1
        with pytest.raises(ValueError, match="num_threads"):
            MosekOptimizationConfig(num_threads=cpu_count + 1)

    def test_mio_max_time_too_low(self):
        """mio_max_time < 1.0 抛出ValueError"""
        with pytest.raises(ValueError, match="mio_max_time"):
            MosekOptimizationConfig(mio_max_time=0.5)

    def test_mio_max_time_too_high(self):
        """mio_max_time > 3600.0 抛出ValueError"""
        with pytest.raises(ValueError, match="mio_max_time"):
            MosekOptimizationConfig(mio_max_time=3601.0)

    def test_max_rounding_attempts_too_low(self):
        """max_rounding_attempts < 1 抛出ValueError"""
        with pytest.raises(ValueError, match="max_rounding_attempts"):
            MosekOptimizationConfig(max_rounding_attempts=0)

    def test_max_rounding_attempts_too_high(self):
        """max_rounding_attempts > 10 抛出ValueError"""
        with pytest.raises(ValueError, match="max_rounding_attempts"):
            MosekOptimizationConfig(max_rounding_attempts=11)

    def test_max_rounded_paths_too_low(self):
        """max_rounded_paths < 1 抛出ValueError"""
        with pytest.raises(ValueError, match="max_rounded_paths"):
            MosekOptimizationConfig(max_rounded_paths=0)

    def test_max_rounded_paths_too_high(self):
        """max_rounded_paths > 20 抛出ValueError"""
        with pytest.raises(ValueError, match="max_rounded_paths"):
            MosekOptimizationConfig(max_rounded_paths=21)

    # --- 序列化/反序列化 ---
    def test_to_dict_from_dict_roundtrip(self):
        """to_dict -> from_dict 往返一致性"""
        config = MosekOptimizationConfig(
            max_paths=3, num_threads=2, mio_max_time=30.0,
            max_rounding_attempts=5, max_rounded_paths=8,
            enable_reduced_paths=False, enable_thread_limit=True,
            enable_mio_time_limit=False, enable_incremental_phi=True,
        )
        data = config.to_dict()
        restored = MosekOptimizationConfig.from_dict(data)
        assert restored.max_paths == config.max_paths
        assert restored.num_threads == config.num_threads
        assert restored.mio_max_time == config.mio_max_time
        assert restored.max_rounding_attempts == config.max_rounding_attempts
        assert restored.max_rounded_paths == config.max_rounded_paths
        assert restored.enable_reduced_paths == config.enable_reduced_paths
        assert restored.enable_thread_limit == config.enable_thread_limit
        assert restored.enable_mio_time_limit == config.enable_mio_time_limit
        assert restored.enable_incremental_phi == config.enable_incremental_phi

    def test_to_dict_contains_all_fields(self):
        """to_dict 包含所有字段"""
        config = MosekOptimizationConfig()
        data = config.to_dict()
        expected_keys = {
            'max_paths', 'num_threads', 'mio_max_time',
            'max_rounding_attempts', 'max_rounded_paths',
            'enable_reduced_paths', 'enable_thread_limit',
            'enable_mio_time_limit', 'enable_incremental_phi',
        }
        assert set(data.keys()) == expected_keys

    # --- summary ---
    def test_summary_not_empty(self):
        """summary() 返回非空字符串"""
        config = MosekOptimizationConfig()
        summary = config.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "MOSEK优化配置" in summary

    def test_summary_contains_effective_values(self):
        """summary() 包含实际生效值"""
        config = MosekOptimizationConfig(max_paths=3, enable_reduced_paths=True)
        summary = config.summary()
        assert "max_paths: 3" in summary

    def test_summary_shows_original_when_disabled(self):
        """summary() 在开关关闭时显示原始值"""
        config = MosekOptimizationConfig(max_paths=3, enable_reduced_paths=False)
        summary = config.summary()
        assert "max_paths: 10" in summary
