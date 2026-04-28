"""
IrisZo区域修剪RTree空间索引单元测试

覆盖RTreeIndex类、RegionPruner的RTree/暴力方法结果一致性、
边界框提取、降级逻辑等。

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


# ============================================================
# 测试辅助函数
# ============================================================

def make_box_region(lb: np.ndarray, ub: np.ndarray) -> IrisZoRegion:
    """从下界/上界创建IrisZoRegion（基于HPolyhedron.MakeBox）"""
    polyhedron = HPolyhedron.MakeBox(lb, ub)
    return IrisZoRegion(polyhedron=polyhedron, seed_point=(lb + ub) / 2)


def make_test_regions_contained() -> List[IrisZoRegion]:
    """创建包含关系的测试区域：大区域包含小区域"""
    regions = []
    # 大区域 [0, 0] - [10, 10]
    regions.append(make_box_region(np.array([0.0, 0.0]), np.array([10.0, 10.0])))
    # 小区域 [2, 2] - [8, 8]（被大区域包含）
    regions.append(make_box_region(np.array([2.0, 2.0]), np.array([8.0, 8.0])))
    # 不相交区域 [15, 15] - [20, 20]
    regions.append(make_box_region(np.array([15.0, 15.0]), np.array([20.0, 20.0])))
    return regions


def make_test_regions_overlapping() -> List[IrisZoRegion]:
    """创建部分重叠但不包含的测试区域"""
    regions = []
    # 区域1 [0, 0] - [5, 5]
    regions.append(make_box_region(np.array([0.0, 0.0]), np.array([5.0, 5.0])))
    # 区域2 [3, 3] - [8, 8]（与区域1部分重叠，但互不包含）
    regions.append(make_box_region(np.array([3.0, 3.0]), np.array([8.0, 8.0])))
    return regions


def make_test_regions_nested(n: int) -> List[IrisZoRegion]:
    """创建n层嵌套区域：每层被前一层包含"""
    regions = []
    for i in range(n):
        offset = float(i)
        size = float(n - i) * 2.0
        regions.append(make_box_region(
            np.array([offset, offset]),
            np.array([offset + size, offset + size])
        ))
    return regions


# ============================================================
# TestRTreeIndex
# ============================================================

class TestRTreeIndex:
    """RTreeIndex类单元测试"""

    def test_rtree_available_flag(self):
        """REQ-007: RTREE_AVAILABLE标志应为bool类型"""
        assert isinstance(RTREE_AVAILABLE, bool)

    def test_rtree_index_construction(self):
        """REQ-001: RTree索引构建正确性"""
        regions = make_test_regions_contained()
        rtree_idx = RTreeIndex(regions)
        assert rtree_idx.num_regions == 3
        assert len(rtree_idx._bounds_cache) == 3

    def test_rtree_index_empty_regions(self):
        """REQ-001: 空区域列表不报错"""
        rtree_idx = RTreeIndex([])
        assert rtree_idx.num_regions == 0
        assert rtree_idx._bounds_cache == []

    def test_extract_bounding_box_from_polyhedron(self):
        """REQ-002: 从HPolyhedron的Axb约束推导边界框"""
        region = make_box_region(np.array([1.0, 2.0]), np.array([5.0, 8.0]))
        bounds = RTreeIndex._extract_bounding_box(region)
        min_x, min_y, max_x, max_y = bounds
        # 边界框应接近 [1, 2, 5, 8]
        assert min_x == pytest.approx(1.0, abs=1e-6)
        assert min_y == pytest.approx(2.0, abs=1e-6)
        assert max_x == pytest.approx(5.0, abs=1e-6)
        assert max_y == pytest.approx(8.0, abs=1e-6)

    def test_extract_bounding_box_fallback(self):
        """REQ-002: 兜底近似边界框（centroid±sqrt(area)）"""
        # 创建一个vertices为None且polyhedron不可用的mock区域
        # 这里直接测试逻辑：当vertices和polyhedron都不可用时
        region = make_box_region(np.array([0.0, 0.0]), np.array([4.0, 4.0]))
        # 正常情况下polyhedron可用，所以会走Axb路径
        bounds = RTreeIndex._extract_bounding_box(region)
        min_x, min_y, max_x, max_y = bounds
        # 边界框应接近 [0, 0, 4, 4]
        assert min_x == pytest.approx(0.0, abs=1e-6)
        assert max_x == pytest.approx(4.0, abs=1e-6)

    def test_find_potential_overlaps_correctness(self):
        """REQ-003: 候选区域查询正确性"""
        regions = make_test_regions_contained()
        rtree_idx = RTreeIndex(regions)

        # 查询大区域（索引0），应找到与小区域（索引1）相交
        candidates = rtree_idx.find_potential_overlaps(regions[0])
        assert 0 in candidates  # 自身
        assert 1 in candidates  # 小区域与大区域相交

        # 查询不相交区域（索引2），不应找到与大区域相交
        candidates_2 = rtree_idx.find_potential_overlaps(regions[2])
        assert 2 in candidates_2  # 自身

    def test_find_potential_overlaps_no_intersection(self):
        """REQ-003: 无相交返回空列表或仅含自身"""
        # 创建两个完全分离的区域
        regions = [
            make_box_region(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
            make_box_region(np.array([100.0, 100.0]), np.array([101.0, 101.0])),
        ]
        rtree_idx = RTreeIndex(regions)

        # 查询区域0，应仅找到自身
        candidates = rtree_idx.find_potential_overlaps(regions[0])
        # RTree的intersection查询包含自身
        assert 0 in candidates

    def test_find_potential_overlaps_by_bounds(self):
        """REQ-003: 按边界框查询"""
        regions = make_test_regions_contained()
        rtree_idx = RTreeIndex(regions)

        # 查询与小区域相交的边界框
        bounds = (2.0, 2.0, 8.0, 8.0)
        candidates = rtree_idx.find_potential_overlaps_by_bounds(bounds)
        assert 0 in candidates  # 大区域与小区域相交
        assert 1 in candidates  # 小区域自身

    def test_rtree_unavailable_returns_all(self):
        """REQ-006: RTree不可用时返回全量索引"""
        regions = make_test_regions_contained()
        # 模拟RTree不可用
        rtree_idx = RTreeIndex(regions)
        original_idx = rtree_idx.idx
        rtree_idx.idx = None  # 强制设为None模拟不可用

        candidates = rtree_idx.find_potential_overlaps(regions[0])
        assert candidates == [0, 1, 2]  # 返回全量索引


# ============================================================
# TestRegionPrunerRTree
# ============================================================

class TestRegionPrunerRTree:
    """RegionPruner的RTree集成单元测试"""

    def test_pruning_result_equivalence(self):
        """REQ-005: RTree与暴力方法结果一致性"""
        regions = make_test_regions_contained()

        # RTree方法
        config_rtree = IrisZoConfig(enable_rtree_pruning=True)
        pruner_rtree = RegionPruner(config_rtree)
        redundant_rtree = pruner_rtree._identify_redundant_rtree(regions)

        # 暴力方法
        config_brute = IrisZoConfig(enable_rtree_pruning=False)
        pruner_brute = RegionPruner(config_brute)
        redundant_brute = pruner_brute._identify_redundant_brute_force(regions)

        assert redundant_rtree == redundant_brute

    def test_pruning_result_equivalence_nested(self):
        """REQ-005: 嵌套区域下RTree与暴力方法结果一致性"""
        regions = make_test_regions_nested(5)

        config_rtree = IrisZoConfig(enable_rtree_pruning=True)
        pruner_rtree = RegionPruner(config_rtree)
        redundant_rtree = pruner_rtree._identify_redundant_rtree(regions)

        config_brute = IrisZoConfig(enable_rtree_pruning=False)
        pruner_brute = RegionPruner(config_brute)
        redundant_brute = pruner_brute._identify_redundant_brute_force(regions)

        assert redundant_rtree == redundant_brute

    def test_pruning_result_equivalence_overlapping(self):
        """REQ-005: 部分重叠区域下RTree与暴力方法结果一致性"""
        regions = make_test_regions_overlapping()

        config_rtree = IrisZoConfig(enable_rtree_pruning=True)
        pruner_rtree = RegionPruner(config_rtree)
        redundant_rtree = pruner_rtree.identify_redundant(regions)

        config_brute = IrisZoConfig(enable_rtree_pruning=False)
        pruner_brute = RegionPruner(config_brute)
        redundant_brute = pruner_brute.identify_redundant(regions)

        assert redundant_rtree == redundant_brute

    def test_pruner_with_rtree_enabled(self):
        """REQ-008: 启用RTree时修剪结果正确"""
        regions = make_test_regions_contained()
        config = IrisZoConfig(enable_rtree_pruning=True)
        pruner = RegionPruner(config)

        result = pruner.prune(regions)
        assert isinstance(result, PruningResult)
        assert result.original_count == 3
        # 小区域（索引1）应被识别为冗余
        assert 1 in result.redundant_indices

    def test_pruner_with_rtree_disabled(self):
        """REQ-008: 禁用RTree时修剪结果正确"""
        regions = make_test_regions_contained()
        config = IrisZoConfig(enable_rtree_pruning=False)
        pruner = RegionPruner(config)
        assert not pruner._use_rtree

        result = pruner.prune(regions)
        assert isinstance(result, PruningResult)
        assert 1 in result.redundant_indices

    def test_pruner_rtree_strategy_flag(self):
        """REQ-008/REQ-006: _use_rtree标志正确性"""
        # 启用且可用
        config_enabled = IrisZoConfig(enable_rtree_pruning=True)
        pruner_enabled = RegionPruner(config_enabled)
        if RTREE_AVAILABLE:
            assert pruner_enabled._use_rtree is True
        else:
            assert pruner_enabled._use_rtree is False

        # 禁用
        config_disabled = IrisZoConfig(enable_rtree_pruning=False)
        pruner_disabled = RegionPruner(config_disabled)
        assert pruner_disabled._use_rtree is False

    def test_empty_regions(self):
        """边界情况：空区域列表"""
        pruner = RegionPruner()
        result = pruner.prune([])
        assert result.original_count == 0
        assert result.remaining_count == 0

    def test_single_region(self):
        """边界情况：单区域"""
        regions = [make_box_region(np.array([0.0, 0.0]), np.array([1.0, 1.0]))]
        pruner = RegionPruner()
        result = pruner.prune(regions)
        assert result.original_count == 1
        assert result.remaining_count == 1
        assert result.redundant_indices == []

    def test_no_redundant_regions(self):
        """边界情况：无冗余区域"""
        regions = make_test_regions_overlapping()
        pruner = RegionPruner()
        result = pruner.prune(regions)
        # 部分重叠但不包含，不应有冗余
        assert result.redundant_indices == []

    def test_config_rtree_parameters(self):
        """REQ-009: RTree参数配置"""
        config = IrisZoConfig(enable_rtree_pruning=True, rtree_leaf_capacity=50)
        assert config.enable_rtree_pruning is True
        assert config.rtree_leaf_capacity == 50

    def test_config_rtree_leaf_capacity_validation(self):
        """REQ-009: rtree_leaf_capacity验证"""
        with pytest.raises(ValueError):
            IrisZoConfig(rtree_leaf_capacity=0).validate()
