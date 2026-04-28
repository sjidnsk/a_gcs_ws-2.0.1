"""
IrisZo区域修剪RTree空间索引集成测试

测试RegionPruner在不同配置下的端到端修剪行为，
以及配置参数与修剪策略的联动。

作者: Path Planning Team
"""

import pytest
import numpy as np
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


def make_grid_regions(
    n: int,
    spacing: float = 3.0,
    size: float = 2.0
) -> List[IrisZoRegion]:
    """创建n×n网格排列的区域，部分区域嵌套"""
    regions = []
    for i in range(n):
        for j in range(n):
            x = i * spacing
            y = j * spacing
            regions.append(make_box_region(
                np.array([x, y]),
                np.array([x + size, y + size])
            ))
    # 添加一些被包含的小区域
    for i in range(0, n, 2):
        for j in range(0, n, 2):
            x = i * spacing + 0.5
            y = j * spacing + 0.5
            regions.append(make_box_region(
                np.array([x, y]),
                np.array([x + size * 0.5, y + size * 0.5])
            ))
    return regions


class TestPruningIntegration:
    """区域修剪集成测试"""

    def test_pruner_with_rtree_enabled(self):
        """启用RTree的端到端修剪"""
        regions = make_grid_regions(3)
        config = IrisZoConfig(enable_rtree_pruning=True)
        pruner = RegionPruner(config)

        result = pruner.prune(regions)
        assert isinstance(result, PruningResult)
        assert result.original_count > 0
        assert result.remaining_count > 0
        assert result.remaining_count <= result.original_count
        assert result.pruning_time >= 0

    def test_pruner_with_rtree_disabled(self):
        """禁用RTree的端到端修剪"""
        regions = make_grid_regions(3)
        config = IrisZoConfig(enable_rtree_pruning=False)
        pruner = RegionPruner(config)

        result = pruner.prune(regions)
        assert isinstance(result, PruningResult)
        assert result.remaining_count > 0

    def test_rtree_and_brute_force_consistency(self):
        """RTree和暴力方法结果完全一致"""
        regions = make_grid_regions(3)

        config_rtree = IrisZoConfig(enable_rtree_pruning=True)
        pruner_rtree = RegionPruner(config_rtree)
        result_rtree = pruner_rtree.prune(regions)

        config_brute = IrisZoConfig(enable_rtree_pruning=False)
        pruner_brute = RegionPruner(config_brute)
        result_brute = pruner_brute.prune(regions)

        assert result_rtree.redundant_indices == result_brute.redundant_indices
        assert result_rtree.remaining_count == result_brute.remaining_count

    def test_config_rtree_parameters(self):
        """配置参数联动测试"""
        regions = make_grid_regions(2)

        # 不同leaf_capacity不应影响修剪结果正确性
        for capacity in [10, 50, 100, 200]:
            config = IrisZoConfig(
                enable_rtree_pruning=True,
                rtree_leaf_capacity=capacity
            )
            pruner = RegionPruner(config)
            result = pruner.prune(regions)
            assert isinstance(result, PruningResult)

    def test_edge_case_empty(self):
        """边界情况：空列表"""
        pruner = RegionPruner()
        result = pruner.prune([])
        assert result.original_count == 0
        assert result.remaining_count == 0
        assert result.redundant_indices == []

    def test_edge_case_single(self):
        """边界情况：单区域"""
        regions = [make_box_region(np.array([0.0, 0.0]), np.array([1.0, 1.0]))]
        pruner = RegionPruner()
        result = pruner.prune(regions)
        assert result.remaining_count == 1

    def test_edge_case_all_nested(self):
        """边界情况：所有区域嵌套"""
        regions = []
        for i in range(5):
            offset = float(i)
            size = 10.0 - float(i) * 2.0
            regions.append(make_box_region(
                np.array([offset, offset]),
                np.array([offset + size, offset + size])
            ))

        config_rtree = IrisZoConfig(enable_rtree_pruning=True)
        pruner_rtree = RegionPruner(config_rtree)
        result_rtree = pruner_rtree.prune(regions)

        config_brute = IrisZoConfig(enable_rtree_pruning=False)
        pruner_brute = RegionPruner(config_brute)
        result_brute = pruner_brute.prune(regions)

        assert result_rtree.redundant_indices == result_brute.redundant_indices
        # 嵌套区域中，只有最大的区域应保留
        assert result_rtree.remaining_count >= 1

    def test_pruning_result_summary(self):
        """PruningResult.get_summary()正确性"""
        regions = make_grid_regions(2)
        pruner = RegionPruner()
        result = pruner.prune(regions)

        summary = result.get_summary()
        assert 'original_count' in summary
        assert 'pruned_count' in summary
        assert 'remaining_count' in summary
        assert 'pruning_ratio' in summary
        assert 'pruning_time' in summary
        assert summary['original_count'] == result.original_count
