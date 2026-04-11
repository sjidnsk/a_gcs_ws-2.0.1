"""
覆盖判定模块

实现基于覆盖半径的覆盖判定逻辑。

作者: Path Planning Team
"""

from typing import List, Tuple, TYPE_CHECKING
import numpy as np

from .iriszo_region_data import IrisZoRegion

if TYPE_CHECKING:
    from ..config.iriszo_config import IrisZoConfig


class CoverageChecker:
    """
    覆盖判定器

    负责基于覆盖半径的覆盖判定，支持单区域覆盖、多区域联合覆盖、
    降级判定等功能。

    Attributes:
        config: 配置参数

    Example:
        >>> from iriszo.config import IrisZoConfig
        >>> config = IrisZoConfig()
        >>> checker = CoverageChecker(config)
        >>> is_covered, reason = checker.check_point_coverage(
        ...     point, effective_radius, regions
        ... )
    """

    def __init__(self, config: "IrisZoConfig"):
        """
        初始化覆盖判定器

        Args:
            config: 配置参数
        """
        self.config = config

    def check_point_coverage(
        self,
        point: np.ndarray,
        effective_radius: float,
        regions: List[IrisZoRegion]
    ) -> Tuple[bool, str]:
        """
        检查点的覆盖情况

        根据有效半径判定点是否被区域集合覆盖。如果有效半径过小，
        降级为点包含判定。

        Args:
            point: 路径点坐标，shape=(2,)
            effective_radius: 有效覆盖半径（米）
            regions: 凸区域列表

        Returns:
            (是否覆盖, 判定原因) 元组

        Example:
            >>> is_covered, reason = checker.check_point_coverage(
            ...     np.array([1.0, 2.0]), 0.5, regions
            ... )
            >>> if is_covered:
            ...     print(f"点被覆盖，原因: {reason}")
        """
        # 降级判定：半径过小
        if effective_radius < self.config.min_effective_radius:
            for i, region in enumerate(regions):
                if region.contains(point):
                    return True, f"downgrade_point_check_by_region_{i}"
            return False, "uncovered_downgrade"

        # 半径判定：检查每个区域
        for i, region in enumerate(regions):
            if self.check_radius_coverage(point, effective_radius, region):
                return True, f"fully_covered_by_region_{i}"

        # 多区域联合覆盖判定（可选）
        if self._check_multi_region_coverage(point, effective_radius, regions):
            return True, "multi_region_covered"

        return False, "uncovered"

    def check_radius_coverage(
        self,
        point: np.ndarray,
        radius: float,
        region: IrisZoRegion
    ) -> bool:
        """
        检查半径范围是否被单个区域包含

        采样圆形边界点，检查所有边界点是否在区域内。
        利用凸集性质：如果边界被包含，则整个圆形区域被包含。

        Args:
            point: 中心点坐标，shape=(2,)
            radius: 覆盖半径（米）
            region: 凸区域

        Returns:
            True如果半径范围被包含，False否则

        Example:
            >>> is_covered = checker.check_radius_coverage(
            ...     np.array([1.0, 2.0]), 0.5, region
            ... )
        """
        # 采样边界点（利用凸集性质）
        boundary_points = self._sample_boundary_points(point, radius)

        # 检查所有边界点是否在区域内
        for bp in boundary_points:
            if not region.contains(bp):
                return False

        return True

    def _sample_boundary_points(
        self,
        point: np.ndarray,
        radius: float,
        num_samples: int = 16
    ) -> np.ndarray:
        """
        采样圆形边界点

        在圆形边界上均匀采样指定数量的点。

        Args:
            point: 中心点坐标，shape=(2,)
            radius: 半径（米）
            num_samples: 采样点数，默认16

        Returns:
            边界点数组，shape=(num_samples, 2)

        Example:
            >>> boundary_points = checker._sample_boundary_points(
            ...     np.array([1.0, 2.0]), 0.5
            ... )
        """
        angles = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
        boundary_points = np.array([
            point + radius * np.array([np.cos(a), np.sin(a)])
            for a in angles
        ])
        return boundary_points

    def _check_multi_region_coverage(
        self,
        point: np.ndarray,
        radius: float,
        regions: List[IrisZoRegion]
    ) -> bool:
        """
        检查多区域联合覆盖

        检查每个边界点是否被至少一个区域包含。

        Args:
            point: 中心点坐标，shape=(2,)
            radius: 覆盖半径（米）
            regions: 凸区域列表

        Returns:
            True如果多区域联合覆盖，False否则

        Note:
            这是简化实现，可能存在边界情况未被覆盖。
        """
        # 采样边界点
        boundary_points = self._sample_boundary_points(point, radius)

        # 检查每个边界点是否被任一区域包含
        for bp in boundary_points:
            if not any(region.contains(bp) for region in regions):
                return False

        return True
