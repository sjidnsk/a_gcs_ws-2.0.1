"""
IrisNp 区域修剪模块

用于移除被其他凸区域完全覆盖的冗余区域。

核心功能：
1. 基于面积覆盖检测冗余区域
2. 使用 RTree 空间索引加速查询
3. 支持批量修剪和增量修剪

作者: Path Planning Team
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import warnings

from .iris_np_region_data import IrisNpRegion

# 尝试导入 rtree
try:
    from rtree import index as rtree_index
    RTREE_AVAILABLE = True
except ImportError:
    RTREE_AVAILABLE = False
    warnings.warn(
        "rtree 库未安装，将使用暴力搜索方法。\n"
        "安装方法: pip install RTree"
    )


@dataclass
class PruningResult:
    """修剪结果"""
    pruned_regions: List[IrisNpRegion]  # 修剪后的区域列表
    removed_indices: List[int]  # 被移除的区域索引
    removed_count: int  # 被移除的区域数量
    pruning_time: float  # 修剪耗时


class RTreeIndex:
    """
    基于 RTree 的区域空间索引

    使用 RTree 库加速区域的空间查询，支持：
    - 快速查找可能重叠的区域
    - 基于边界框的空间查询
    """

    def __init__(self, regions: List[IrisNpRegion]):
        """
        初始化 RTree 索引

        Args:
            regions: 区域列表
        """
        self.regions = regions
        self.num_regions = len(regions)

        if not RTREE_AVAILABLE:
            self.idx = None
            return

        # 创建 RTree 索引
        # properties=rtree_index.Property() 可以设置索引属性
        self.idx = rtree_index.Index()

        # 添加每个区域的边界框
        for i, region in enumerate(regions):
            x_min, y_min = region.vertices.min(axis=0)
            x_max, y_max = region.vertices.max(axis=0)

            # RTree 使用 (left, bottom, right, top) 格式
            self.idx.insert(i, (x_min, y_min, x_max, y_max))

    def find_potential_overlaps(self, region: IrisNpRegion) -> List[int]:
        """
        查找可能与指定区域重叠的区域索引

        Args:
            region: 查询区域

        Returns:
            可能重叠的区域索引列表
        """
        if self.idx is None:
            # 如果 RTree 不可用，返回所有索引
            return list(range(self.num_regions))

        x_min, y_min = region.vertices.min(axis=0)
        x_max, y_max = region.vertices.max(axis=0)

        # 查询相交的边界框
        overlaps = list(self.idx.intersection((x_min, y_min, x_max, y_max)))

        return overlaps


class RegionPruner:
    """
    区域修剪器

    检测并移除被其他区域完全覆盖的冗余区域。

    算法流程：
    1. 构建 RTree 空间索引
    2. 对每个区域，查找可能重叠的候选区域
    3. 使用面积覆盖检测判断是否被完全覆盖
    4. 移除被覆盖的区域

    Attributes:
        verbose: 是否输出详细信息
        use_rtree: 是否使用 RTree 索引（如果可用）
        sample_resolution: 面积覆盖检测的采样分辨率（米）
    """

    def __init__(
        self,
        verbose: bool = False,
        use_rtree: bool = True,
        sample_resolution: float = 0.05
    ):
        """
        初始化区域修剪器

        Args:
            verbose: 是否输出详细信息
            use_rtree: 是否使用 RTree 索引（如果可用）
            sample_resolution: 面积覆盖检测的采样分辨率（米）
                             较小的值提供更精确的检测，但计算更慢
        """
        self.verbose = verbose
        self.use_rtree = use_rtree and RTREE_AVAILABLE
        self.sample_resolution = sample_resolution

        if self.verbose:
            print(f"区域修剪器初始化:")
            print(f"  - RTree 索引: {'启用' if self.use_rtree else '禁用'}")
            print(f"  - 采样分辨率: {self.sample_resolution} 米")

    def prune(self, regions: List[IrisNpRegion]) -> PruningResult:
        """
        修剪区域列表，移除被完全覆盖的区域

        Args:
            regions: 区域列表

        Returns:
            PruningResult: 修剪结果，包含：
                - pruned_regions: 修剪后的区域列表
                - removed_indices: 被移除的区域索引
                - removed_count: 被移除的区域数量
                - pruning_time: 修剪耗时
        """
        import time

        if self.verbose:
            print("\n" + "="*70)
            print("开始区域修剪")
            print("="*70)
            print(f"输入区域数量: {len(regions)}")

        start_time = time.time()

        if len(regions) <= 1:
            # 只有一个或没有区域，无需修剪
            if self.verbose:
                print("区域数量 <= 1，无需修剪")

            return PruningResult(
                pruned_regions=regions.copy(),
                removed_indices=[],
                removed_count=0,
                pruning_time=time.time() - start_time
            )

        # 构建 RTree 索引
        if self.use_rtree:
            rtree = RTreeIndex(regions)
        else:
            rtree = None

        # 检测被覆盖的区域
        covered_indices = set()

        for i, region in enumerate(regions):
            if i in covered_indices:
                continue

            # 查找可能重叠的候选区域
            if rtree is not None:
                candidate_indices = rtree.find_potential_overlaps(region)
            else:
                candidate_indices = list(range(len(regions)))

            # 排除自己和已经标记为被覆盖的区域
            candidate_indices = [
                j for j in candidate_indices
                if j != i and j not in covered_indices
            ]

            if len(candidate_indices) == 0:
                continue

            # 检查是否被其他区域的并集覆盖
            is_covered = self._is_region_covered(
                region, regions, candidate_indices
            )

            if is_covered:
                covered_indices.add(i)
                if self.verbose:
                    print(f"  区域 {i} 被其他区域完全覆盖，将被移除")

        # 移除被覆盖的区域
        pruned_regions = [
            region for i, region in enumerate(regions)
            if i not in covered_indices
        ]

        removed_indices = sorted(list(covered_indices))
        pruning_time = time.time() - start_time

        if self.verbose:
            print(f"\n修剪完成:")
            print(f"  - 移除区域数量: {len(removed_indices)}")
            print(f"  - 保留区域数量: {len(pruned_regions)}")
            print(f"  - 修剪耗时: {pruning_time:.4f} 秒")

        return PruningResult(
            pruned_regions=pruned_regions,
            removed_indices=removed_indices,
            removed_count=len(removed_indices),
            pruning_time=pruning_time
        )

    def _is_region_covered(
        self,
        region: IrisNpRegion,
        all_regions: List[IrisNpRegion],
        candidate_indices: List[int]
    ) -> bool:
        """
        检查区域是否被其他区域的并集完全覆盖

        使用采样方法检测面积覆盖：
        1. 在区域内均匀采样点
        2. 检查每个采样点是否至少在一个候选区域内
        3. 如果所有采样点都被覆盖，则认为区域被完全覆盖

        Args:
            region: 待检查的区域
            all_regions: 所有区域列表
            candidate_indices: 候选覆盖区域的索引列表

        Returns:
            True 如果区域被完全覆盖
        """
        # 获取候选区域
        candidate_regions = [all_regions[i] for i in candidate_indices]

        # 在区域内采样点
        sample_points = self._sample_points_in_region(region)

        if len(sample_points) == 0:
            # 无法采样，保守地认为不被覆盖
            return False

        # 检查每个采样点是否至少在一个候选区域内
        for point in sample_points:
            is_point_covered = any(
                candidate_region.contains(point, tol=1e-6)
                for candidate_region in candidate_regions
            )

            if not is_point_covered:
                # 发现未覆盖的点，区域未被完全覆盖
                return False

        # 所有采样点都被覆盖
        return True

    def _sample_points_in_region(
        self,
        region: IrisNpRegion,
        num_samples: int = 100
    ) -> List[np.ndarray]:
        """
        在区域内均匀采样点

        使用拒绝采样方法：
        1. 在区域的边界框内均匀采样
        2. 检查采样点是否在区域内
        3. 保留在区域内的点

        Args:
            region: 区域
            num_samples: 目标采样点数量

        Returns:
            采样点列表
        """
        # 获取边界框
        x_min, y_min = region.vertices.min(axis=0)
        x_max, y_max = region.vertices.max(axis=0)

        sample_points = []
        max_attempts = num_samples * 10  # 防止无限循环
        attempts = 0

        while len(sample_points) < num_samples and attempts < max_attempts:
            attempts += 1

            # 在边界框内随机采样
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            point = np.array([x, y])

            # 检查点是否在区域内
            if region.contains(point, tol=1e-6):
                sample_points.append(point)

        return sample_points


def prune_regions(
    regions: List[IrisNpRegion],
    verbose: bool = False,
    use_rtree: bool = True,
    sample_resolution: float = 0.05
) -> PruningResult:
    """
    修剪区域列表，移除被完全覆盖的区域（便捷函数）

    Args:
        regions: 区域列表
        verbose: 是否输出详细信息
        use_rtree: 是否使用 RTree 索引（如果可用）
        sample_resolution: 面积覆盖检测的采样分辨率（米）

    Returns:
        PruningResult: 修剪结果

    Example:
        >>> from src.iris_pkg.core.iris_np_region_pruner import prune_regions
        >>> from src.iris_pkg.core.iris_np_region_data import IrisNpRegion
        >>> result = prune_regions(regions, verbose=True)
        >>> print(f"移除了 {result.removed_count} 个冗余区域")
        >>> pruned_regions = result.pruned_regions
    """
    pruner = RegionPruner(
        verbose=verbose,
        use_rtree=use_rtree,
        sample_resolution=sample_resolution
    )
    return pruner.prune(regions)
