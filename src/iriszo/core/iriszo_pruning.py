"""
区域修剪模块

实现区域修剪功能，过滤掉被更大凸区域完全覆盖的凸区域。

作者: Path Planning Team
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import time

from .iriszo_region_data import IrisZoRegion
from ..config.iriszo_config import IrisZoConfig


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


class RegionPruner:
    """
    区域修剪器主类

    该类提供区域修剪功能，识别并移除被更大区域完全覆盖的冗余区域。

    Attributes:
        config: IrisZo配置对象

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

        检查每对区域之间的包含关系，识别被其他区域完全覆盖的区域。

        Args:
            regions: 凸区域列表

        Returns:
            冗余区域索引列表（已排序）

        Example:
            >>> redundant = pruner.identify_redundant(regions)
            >>> print(f"冗余区域索引: {redundant}")
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
