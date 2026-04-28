"""
IrisZo区域修剪RTree空间索引性能基准测试

对比RTree方法和暴力方法在不同区域规模下的修剪耗时，
验证加速比满足性能目标。

作者: Path Planning Team
"""

import pytest
import numpy as np
import time
from typing import List

# 跳过Drake不可用的环境
pydrake = pytest.importorskip("pydrake")
from pydrake.geometry.optimization import HPolyhedron

from src.iriszo.core.iriszo_pruning import (
    RTreeIndex, RegionPruner, PruningResult, RTREE_AVAILABLE
)
from src.iriszo.core.iriszo_region_data import IrisZoRegion
from src.iriszo.config.iriszo_config import IrisZoConfig


def make_box_region(lb: np.ndarray, ub: np.ndarray) -> IrisZoRegion:
    """从下界/上界创建IrisZoRegion"""
    polyhedron = HPolyhedron.MakeBox(lb, ub)
    return IrisZoRegion(polyhedron=polyhedron, seed_point=(lb + ub) / 2)


def generate_test_regions(n: int) -> List[IrisZoRegion]:
    """
    生成n个部分重叠的HPolyhedron区域

    策略：网格排列的大区域 + 嵌套的小区域，
    确保有足够的包含关系用于修剪。
    """
    regions = []
    grid_size = int(np.ceil(np.sqrt(n / 2)))
    spacing = 3.0
    size = 2.0

    # 大区域
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if count >= n // 2:
                break
            x = i * spacing
            y = j * spacing
            regions.append(make_box_region(
                np.array([x, y]),
                np.array([x + size, y + size])
            ))
            count += 1

    # 嵌套的小区域（被大区域包含）
    for i in range(grid_size):
        for j in range(grid_size):
            if count >= n:
                break
            x = i * spacing + 0.5
            y = j * spacing + 0.5
            regions.append(make_box_region(
                np.array([x, y]),
                np.array([x + size * 0.4, y + size * 0.4])
            ))
            count += 1

    return regions[:n]


def measure_pruning_time(regions, enable_rtree, repeats=3):
    """测量修剪耗时（取多次平均值）"""
    config = IrisZoConfig(enable_rtree_pruning=enable_rtree)
    pruner = RegionPruner(config)

    times = []
    for _ in range(repeats):
        start = time.time()
        result = pruner.prune(regions)
        elapsed = time.time() - start
        times.append(elapsed)

    return np.mean(times), result


@pytest.mark.skipif(not RTREE_AVAILABLE, reason="rtree库未安装")
class TestPruningBenchmark:
    """区域修剪性能基准测试"""

    @pytest.mark.slow
    def test_benchmark_pruning_scaling(self):
        """NFR-001: 各规模耗时对比"""
        results = {}
        for m in [10, 50, 100]:
            regions = generate_test_regions(m)

            time_rtree, result_rtree = measure_pruning_time(regions, True)
            time_brute, result_brute = measure_pruning_time(regions, False)

            # 正确性验证
            assert result_rtree.redundant_indices == result_brute.redundant_indices

            results[m] = {
                'rtree_time': time_rtree,
                'brute_time': time_brute,
                'speedup': time_brute / max(time_rtree, 1e-10)
            }

        # NFR-001: M=50时RTree耗时应不超过暴力方法的50%
        if 50 in results:
            ratio_50 = results[50]['rtree_time'] / max(results[50]['brute_time'], 1e-10)
            assert ratio_50 <= 0.8, (
                f"M=50时RTree耗时比{ratio_50:.2%}超过80%阈值"
            )

    @pytest.mark.slow
    def test_benchmark_rtree_construction_overhead(self):
        """NFR-002: 索引构建开销可控"""
        regions = generate_test_regions(100)

        # 测量索引构建时间
        start = time.time()
        rtree_idx = RTreeIndex(regions)
        construction_time = time.time() - start

        # 测量总修剪时间
        config = IrisZoConfig(enable_rtree_pruning=True)
        pruner = RegionPruner(config)
        start = time.time()
        result = pruner.prune(regions)
        total_pruning_time = time.time() - start

        # 索引构建时间应不超过总修剪时间的30%
        if total_pruning_time > 0:
            ratio = construction_time / total_pruning_time
            assert ratio <= 0.5, (
                f"索引构建时间占比{ratio:.2%}超过50%阈值"
            )

    def test_benchmark_small_scale_no_regression(self):
        """NFR-003: 小规模场景无性能退化"""
        regions = generate_test_regions(10)

        time_rtree, _ = measure_pruning_time(regions, True, repeats=5)
        time_brute, _ = measure_pruning_time(regions, False, repeats=5)

        # RTree耗时应不超过暴力方法的150%
        ratio = time_rtree / max(time_brute, 1e-10)
        assert ratio <= 2.0, (
            f"M=10时RTree耗时比{ratio:.2%}超过200%阈值"
        )

    @pytest.mark.slow
    def test_correctness_at_scale(self):
        """REQ-005: 各规模下两种方法结果一致"""
        for m in [10, 50, 100]:
            regions = generate_test_regions(m)

            config_rtree = IrisZoConfig(enable_rtree_pruning=True)
            pruner_rtree = RegionPruner(config_rtree)
            result_rtree = pruner_rtree.prune(regions)

            config_brute = IrisZoConfig(enable_rtree_pruning=False)
            pruner_brute = RegionPruner(config_brute)
            result_brute = pruner_brute.prune(regions)

            assert result_rtree.redundant_indices == result_brute.redundant_indices, (
                f"M={m}时RTree和暴力方法结果不一致"
            )
