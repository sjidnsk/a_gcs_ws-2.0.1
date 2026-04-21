"""
区域修剪模块

实现区域修剪功能，过滤掉被更大凸区域完全覆盖的凸区域。
支持RTree空间索引加速，将冗余区域检测从O(M²)降至O(M log M)。

作者: Path Planning Team
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import time
import warnings

from .iriszo_region_data import IrisZoRegion
from ..config.iriszo_config import IrisZoConfig

# rtree库可用性检测
try:
    from rtree import index as rtree_index
    RTREE_AVAILABLE = True
except ImportError:
    RTREE_AVAILABLE = False
    rtree_index = None


@dataclass
class PruningResult:
    """
    区域修剪结果数据结构

    该类表示区域修剪的完整结果，包含原始区域列表、修剪后区域列表、
    冗余区域索引等信息。

    Attributes:
        original_regions: 原始区域列表
        pruned_regions: 修剪后的区域列表
        original_count: 原始区域数量
        pruned_count: 被修剪的区域数量
        remaining_count: 剩余区域数量
        pruning_ratio: 修剪比例
        redundant_indices: 冗余区域索引列表
        pruning_time: 修剪耗时（秒）

    Example:
        >>> result = pruner.prune(regions)
        >>> print(f"修剪了 {result.pruned_count} 个冗余区域")
        >>> print(f"修剪比例: {result.pruning_ratio:.2%}")
    """

    original_regions: List[IrisZoRegion] = field(default_factory=list)
    pruned_regions: List[IrisZoRegion] = field(default_factory=list)
    original_count: int = 0
    pruned_count: int = 0
    remaining_count: int = 0
    pruning_ratio: float = 0.0
    redundant_indices: List[int] = field(default_factory=list)
    pruning_time: float = 0.0

    def __post_init__(self):
        """
        初始化后处理

        自动计算统计信息。
        """
        self.original_count = len(self.original_regions)
        self.remaining_count = len(self.pruned_regions)
        self.pruned_count = self.original_count - self.remaining_count
        if self.original_count > 0:
            self.pruning_ratio = self.pruned_count / self.original_count

    def get_summary(self) -> dict:
        """
        获取摘要信息

        Returns:
            包含关键统计信息的字典

        Example:
            >>> summary = result.get_summary()
            >>> print(f"修剪比例: {summary['pruning_ratio']:.2%}")
        """
        return {
            'original_count': self.original_count,
            'pruned_count': self.pruned_count,
            'remaining_count': self.remaining_count,
            'pruning_ratio': self.pruning_ratio,
            'pruning_time': self.pruning_time
        }


class RTreeIndex:
    """
    基于RTree的区域空间索引

    使用RTree库加速区域的空间查询，支持：
    - 快速查找边界框相交的候选区域
    - 基于边界框的空间查询
    - rtree不可用时回退到全量索引

    边界框提取采用三级策略：
    1. vertices可用时 → 直接min/max计算
    2. polyhedron可用时 → 从Axb约束推导边界框
    3. 兜底 → centroid±sqrt(area)近似

    Attributes:
        regions: 区域列表
        num_regions: 区域数量
        idx: RTree索引实例（rtree不可用时为None）
        _bounds_cache: 边界框缓存列表

    Example:
        >>> rtree_idx = RTreeIndex(regions)
        >>> candidates = rtree_idx.find_potential_overlaps(query_region)
    """

    def __init__(
        self,
        regions: List[IrisZoRegion],
        leaf_capacity: int = 100
    ) -> None:
        """
        初始化RTree索引

        Args:
            regions: 区域列表
            leaf_capacity: RTree叶节点容量（默认100）
        """
        self.regions = regions
        self.num_regions = len(regions)
        self._bounds_cache: List[Tuple[float, ...]] = []

        if not RTREE_AVAILABLE or not regions:
            self.idx = None
            # 仍需计算边界框缓存（供find_potential_overlaps_by_bounds使用）
            for i, region in enumerate(regions):
                bounds = self._extract_bounding_box(region)
                self._bounds_cache.append(bounds)
            return

        # 创建RTree索引
        properties = rtree_index.Property()
        properties.leaf_capacity = leaf_capacity
        self.idx = rtree_index.Index(properties=properties)

        # 添加每个区域的边界框
        for i, region in enumerate(regions):
            bounds = self._extract_bounding_box(region)
            self._bounds_cache.append(bounds)
            # RTree使用(min_x, min_y, max_x, max_y)格式
            self.idx.insert(i, bounds)

    def find_potential_overlaps(
        self,
        region: IrisZoRegion
    ) -> List[int]:
        """
        查找可能与指定区域重叠的区域索引

        Args:
            region: 查询区域

        Returns:
            可能重叠的区域索引列表（RTree不可用时返回全量索引）
        """
        bounds = self._extract_bounding_box(region)
        return self.find_potential_overlaps_by_bounds(bounds)

    def find_potential_overlaps_by_bounds(
        self,
        bounds: Tuple[float, ...]
    ) -> List[int]:
        """
        根据边界框查询可能重叠的区域索引

        Args:
            bounds: 边界框 (min_x, min_y, max_x, max_y)

        Returns:
            可能重叠的区域索引列表
        """
        if self.idx is None:
            return list(range(self.num_regions))

        return list(self.idx.intersection(bounds))

    @staticmethod
    def _extract_bounding_box(
        region: IrisZoRegion
    ) -> Tuple[float, float, float, float]:
        """
        从IrisZoRegion提取边界框

        提取策略（按优先级）：
        1. 若vertices可用且非空 → min/max计算
        2. 若polyhedron可用 → 从Axb约束推导边界框
        3. 兜底 → centroid±sqrt(area)近似

        Args:
            region: 区域

        Returns:
            (min_x, min_y, max_x, max_y)
        """
        # 第一级：vertices可用
        if region.vertices is not None and len(region.vertices) > 0:
            min_coords = np.min(region.vertices, axis=0)
            max_coords = np.max(region.vertices, axis=0)
            return (min_coords[0], min_coords[1], max_coords[0], max_coords[1])

        # 第二级：从Axb约束推导边界框
        try:
            if region.polyhedron is not None:
                A = region.polyhedron.A()
                b = region.polyhedron.b()
                dim = A.shape[1]

                # 向量化边界框计算（与iriszo_sampler._get_bounding_box一致）
                pos_mask = A > 1e-10
                ratios_pos = np.where(pos_mask, b[:, None] / A, np.inf)
                ub = np.min(ratios_pos, axis=0)

                neg_mask = A < -1e-10
                ratios_neg = np.where(neg_mask, b[:, None] / A, -np.inf)
                lb = np.max(ratios_neg, axis=0)

                # 检查是否有效（非inf）
                if np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)):
                    return (lb[0], lb[1], ub[0], ub[1])
        except Exception:
            pass

        # 第三级兜底：centroid±sqrt(area)近似
        radius = np.sqrt(max(region.area, 0.0))
        cx, cy = region.centroid[0], region.centroid[1]
        return (cx - radius, cy - radius, cx + radius, cy + radius)


class RegionPruner:
    """
    区域修剪器主类

    该类提供区域修剪功能，识别并移除被更大区域完全覆盖的冗余区域。

    当RTree可用且启用时，使用空间索引加速候选区域查找，
    将冗余检测从O(M²)降至O(M log M)；否则回退到双重循环实现。

    Attributes:
        config: IrisZo配置对象
        _use_rtree: 是否使用RTree索引

    Example:
        >>> pruner = RegionPruner()
        >>> result = pruner.prune(regions)
        >>> print(f"剩余 {result.remaining_count} 个区域")
    """

    def __init__(self, config: Optional[IrisZoConfig] = None):
        """
        初始化区域修剪器

        Args:
            config: IrisZo配置对象，可选
        """
        self.config = config or IrisZoConfig()
        self._use_rtree = (
            self.config.enable_rtree_pruning
            and RTREE_AVAILABLE
        )

    def prune(
        self,
        regions: List[IrisZoRegion]
    ) -> PruningResult:
        """
        修剪冗余区域

        识别并移除被更大区域完全覆盖的冗余区域。

        Args:
            regions: 凸区域列表

        Returns:
            PruningResult对象，包含修剪结果

        Example:
            >>> result = pruner.prune(regions)
            >>> print(f"修剪了 {result.pruned_count} 个区域")
        """
        start_time = time.time()

        if not regions:
            return PruningResult(pruning_time=time.time() - start_time)

        # 识别冗余区域
        redundant_indices = self.identify_redundant(regions)

        # 过滤冗余区域
        pruned_regions = [
            r for i, r in enumerate(regions)
            if i not in redundant_indices
        ]

        # 构建结果
        result = PruningResult(
            original_regions=regions,
            pruned_regions=pruned_regions,
            redundant_indices=redundant_indices,
            pruning_time=time.time() - start_time
        )

        return result

    def identify_redundant(
        self,
        regions: List[IrisZoRegion]
    ) -> List[int]:
        """
        识别冗余区域的索引

        当RTree可用且启用时，使用空间索引加速候选区域查找；
        否则回退到双重循环实现。

        Args:
            regions: 凸区域列表

        Returns:
            冗余区域索引列表（已排序）

        Example:
            >>> redundant = pruner.identify_redundant(regions)
            >>> print(f"冗余区域索引: {redundant}")
        """
        if self._use_rtree:
            return self._identify_redundant_rtree(regions)
        else:
            return self._identify_redundant_brute_force(regions)

    def _identify_redundant_brute_force(
        self,
        regions: List[IrisZoRegion]
    ) -> List[int]:
        """
        使用双重循环识别冗余区域（O(M²)暴力方法）

        Args:
            regions: 凸区域列表

        Returns:
            冗余区域索引列表（已排序）
        """
        redundant = set()

        for i, region_i in enumerate(regions):
            if i in redundant:
                continue

            for j, region_j in enumerate(regions):
                if i == j or j in redundant:
                    continue

                # 检查包含关系
                if region_i.area < region_j.area:
                    # region_i更小，检查是否被region_j包含
                    if self.check_containment(region_j, region_i):
                        redundant.add(i)
                        break
                elif region_i.area > region_j.area:
                    # region_i更大，检查是否包含region_j
                    if self.check_containment(region_i, region_j):
                        redundant.add(j)

        return sorted(list(redundant))

    def _identify_redundant_rtree(
        self,
        regions: List[IrisZoRegion]
    ) -> List[int]:
        """
        使用RTree空间索引识别冗余区域（O(M log M)加速方法）

        通过RTree查询候选区域替代全量遍历，仅对边界框相交的
        区域对执行精确的包含关系检查。

        Args:
            regions: 凸区域列表

        Returns:
            冗余区域索引列表（已排序）
        """
        redundant = set()

        # 构建RTree索引
        rtree_index_obj = RTreeIndex(
            regions,
            leaf_capacity=self.config.rtree_leaf_capacity
        )

        for i, region_i in enumerate(regions):
            if i in redundant:
                continue

            # 通过RTree查找候选区域
            candidate_indices = rtree_index_obj.find_potential_overlaps(region_i)

            for j in candidate_indices:
                if i == j or j in redundant:
                    continue

                region_j = regions[j]

                # 检查包含关系（与暴力方法逻辑一致）
                if region_i.area < region_j.area:
                    # region_i更小，检查是否被region_j包含
                    if self.check_containment(region_j, region_i):
                        redundant.add(i)
                        break
                elif region_i.area > region_j.area:
                    # region_i更大，检查是否包含region_j
                    if self.check_containment(region_i, region_j):
                        redundant.add(j)

        return sorted(list(redundant))

    def check_containment(
        self,
        region_a: IrisZoRegion,
        region_b: IrisZoRegion
    ) -> bool:
        """
        检查region_a是否完全包含region_b（优化版本）

        使用多种方法进行包含关系判断：
        1. 预筛选：体积比较
        2. 预筛选：中心检查
        3. 顶点检查法（如果可获取顶点）
        4. 采样近似法（作为备选）

        Args:
            region_a: 外部区域
            region_b: 内部区域

        Returns:
            True如果region_a完全包含region_b，False否则

        Example:
            >>> if pruner.check_containment(outer, inner):
            ...     print("outer包含inner")
        """
        # 预筛选：体积比较
        if region_b.area > region_a.area:
            return False

        # 预筛选：中心检查
        if not region_a.contains(region_b.centroid):
            return False

        # 方法1：顶点检查法（如果可获取顶点）
        if region_b.vertices is not None and len(region_b.vertices) > 0:
            for vertex in region_b.vertices:
                if not region_a.contains(vertex):
                    return False
            return True

        # 方法2：采样近似法
        try:
            samples = self._sample_region(region_b, num_samples=50)
            for point in samples:
                if not region_a.contains(point):
                    return False
            return True
        except Exception:
            return False

    def _sample_region(
        self,
        region: IrisZoRegion,
        num_samples: int = 20
    ) -> np.ndarray:
        """
        在区域内采样点（内部方法）

        使用Hit-and-Run采样方法在区域内生成均匀分布的采样点。

        Args:
            region: 凸区域
            num_samples: 采样点数量

        Returns:
            采样点数组，shape=(num_samples, dim)

        Raises:
            ImportError: 如果采样器不可用
        """
        # 使用Hit-and-Run采样
        from .iriszo_sampler import HitAndRunSampler
        sampler = HitAndRunSampler(self.config)
        return sampler.sample(region.polyhedron, num_samples)
