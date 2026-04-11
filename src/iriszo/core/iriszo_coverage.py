"""
覆盖验证模块

实现路径覆盖验证功能，计算路径被生成的凸区域覆盖的比例。

作者: Path Planning Team
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import time

from .iriszo_region_data import IrisZoRegion
from ..config.iriszo_config import IrisZoConfig


@dataclass
class CoverageResult:
    """
    覆盖验证结果数据结构

    该类表示路径覆盖验证的完整结果，包含覆盖率、覆盖点索引、
    未覆盖点索引、未覆盖段等信息。

    Attributes:
        coverage_ratio: 覆盖率，范围[0.0, 1.0]
        covered_indices: 被覆盖的路径点索引列表
        uncovered_indices: 未被覆盖的路径点索引列表
        uncovered_segments: 未覆盖的连续段列表，每个元素为(start_idx, end_idx)
        validation_time: 验证耗时（秒）
        total_points: 路径点总数
        covered_points: 被覆盖的点数
        uncovered_points: 未被覆盖的点数

    Example:
        >>> result = validator.validate(path, regions)
        >>> print(f"覆盖率: {result.coverage_ratio:.2%}")
        >>> print(f"未覆盖段数: {len(result.uncovered_segments)}")
    """

    coverage_ratio: float = 0.0
    covered_indices: List[int] = field(default_factory=list)
    uncovered_indices: List[int] = field(default_factory=list)
    uncovered_segments: List[Tuple[int, int]] = field(default_factory=list)
    validation_time: float = 0.0
    total_points: int = 0
    covered_points: int = 0
    uncovered_points: int = 0

    def is_fully_covered(self) -> bool:
        """
        检查是否完全覆盖

        Returns:
            True如果覆盖率为100%，False否则

        Example:
            >>> if result.is_fully_covered():
            ...     print("路径完全被覆盖")
        """
        return self.coverage_ratio == 1.0

    def get_summary(self) -> dict:
        """
        获取摘要信息

        Returns:
            包含关键统计信息的字典

        Example:
            >>> summary = result.get_summary()
            >>> print(f"覆盖率: {summary['coverage_ratio']:.2%}")
        """
        return {
            'coverage_ratio': self.coverage_ratio,
            'total_points': self.total_points,
            'covered_points': self.covered_points,
            'uncovered_points': self.uncovered_points,
            'num_uncovered_segments': len(self.uncovered_segments),
            'validation_time': self.validation_time
        }


class CoverageValidator:
    """
    覆盖验证器主类

    该类提供路径覆盖验证功能，检查路径点是否被凸区域集合覆盖。

    Attributes:
        config: IrisZo配置对象

    Example:
        >>> validator = CoverageValidator()
        >>> result = validator.validate(path, regions)
        >>> print(f"覆盖率: {result.coverage_ratio:.2%}")
    """

    def __init__(self, config: Optional[IrisZoConfig] = None):
        """
        初始化覆盖验证器

        Args:
            config: IrisZo配置对象，可选
        """
        self.config = config or IrisZoConfig()

    def validate(
        self,
        path: np.ndarray,
        regions: List[IrisZoRegion]
    ) -> CoverageResult:
        """
        验证路径覆盖情况

        检查路径中的每个点是否被至少一个凸区域覆盖，计算覆盖率，
        并分析未覆盖段的连续性。

        Args:
            path: 路径点数组，shape=(N, dim)
            regions: 凸区域列表

        Returns:
            CoverageResult对象，包含完整的覆盖验证结果

        Example:
            >>> path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
            >>> result = validator.validate(path, regions)
            >>> print(f"覆盖率: {result.coverage_ratio:.2%}")
        """
        start_time = time.time()

        # 输入验证
        if len(path) == 0 or len(regions) == 0:
            return CoverageResult(validation_time=time.time() - start_time)

        # 初始化结果
        covered_indices = []
        uncovered_indices = []

        # 遍历路径点
        for i, point in enumerate(path):
            if self.check_point_coverage(point, regions):
                covered_indices.append(i)
            else:
                uncovered_indices.append(i)

        # 计算覆盖率
        coverage_ratio = len(covered_indices) / len(path)

        # 分析覆盖连续性
        uncovered_segments = self._analyze_continuity(uncovered_indices)

        # 构建结果
        result = CoverageResult(
            coverage_ratio=coverage_ratio,
            covered_indices=covered_indices,
            uncovered_indices=uncovered_indices,
            uncovered_segments=uncovered_segments,
            validation_time=time.time() - start_time,
            total_points=len(path),
            covered_points=len(covered_indices),
            uncovered_points=len(uncovered_indices)
        )

        return result

    def check_point_coverage(
        self,
        point: np.ndarray,
        regions: List[IrisZoRegion]
    ) -> bool:
        """
        检查单点是否被区域集合覆盖

        Args:
            point: 待检查的点，shape=(dim,)
            regions: 凸区域列表

        Returns:
            True如果点被至少一个区域覆盖，False否则

        Example:
            >>> point = np.array([0.5, 0.5])
            >>> if validator.check_point_coverage(point, regions):
            ...     print("点被覆盖")
        """
        for region in regions:
            if region.contains(point):
                return True
        return False

    def get_coverage_indices(
        self,
        path: np.ndarray,
        regions: List[IrisZoRegion]
    ) -> Tuple[List[int], List[int]]:
        """
        获取覆盖和未覆盖点的索引

        Args:
            path: 路径点数组，shape=(N, dim)
            regions: 凸区域列表

        Returns:
            (covered_indices, uncovered_indices) 元组

        Example:
            >>> covered, uncovered = validator.get_coverage_indices(path, regions)
            >>> print(f"覆盖点数: {len(covered)}")
        """
        covered = []
        uncovered = []

        for i, point in enumerate(path):
            if self.check_point_coverage(point, regions):
                covered.append(i)
            else:
                uncovered.append(i)

        return covered, uncovered

    def analyze_coverage_continuity(
        self,
        path: np.ndarray,
        regions: List[IrisZoRegion]
    ) -> List[Tuple[int, int]]:
        """
        分析覆盖连续性

        识别路径中未被覆盖的连续段。

        Args:
            path: 路径点数组，shape=(N, dim)
            regions: 凸区域列表

        Returns:
            未覆盖连续段列表，每个元素为(start_idx, end_idx)

        Example:
            >>> segments = validator.analyze_coverage_continuity(path, regions)
            >>> for start, end in segments:
            ...     print(f"未覆盖段: [{start}, {end}]")
        """
        _, uncovered = self.get_coverage_indices(path, regions)
        return self._analyze_continuity(uncovered)

    def _analyze_continuity(
        self,
        uncovered_indices: List[int]
    ) -> List[Tuple[int, int]]:
        """
        分析未覆盖段的连续性（内部方法）

        将连续的未覆盖索引合并为段。

        Args:
            uncovered_indices: 未覆盖点索引列表

        Returns:
            未覆盖连续段列表，每个元素为(start_idx, end_idx)
        """
        if not uncovered_indices:
            return []

        segments = []
        start = uncovered_indices[0]
        prev = start

        for idx in uncovered_indices[1:]:
            if idx != prev + 1:
                segments.append((start, prev))
                start = idx
            prev = idx

        segments.append((start, prev))
        return segments
