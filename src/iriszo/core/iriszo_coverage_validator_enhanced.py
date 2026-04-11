"""
增强版覆盖验证器

集成所有子模块，实现基于覆盖半径的完整覆盖验证流程。

作者: Path Planning Team
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import time
import logging

from .iriszo_region_data import IrisZoRegion
from .iriszo_coverage_radius import RadiusCalculator
from .iriszo_obstacle_detector import ObstacleDetector
from .iriszo_coverage_checker import CoverageChecker
from ..config.iriszo_config import IrisZoConfig


@dataclass
class EnhancedCoverageResult:
    """
    增强版覆盖验证结果数据结构

    包含更详细的覆盖验证信息。

    Attributes:
        coverage_ratio: 覆盖率，范围[0.0, 1.0]
        covered_indices: 被覆盖的路径点索引列表
        uncovered_indices: 未被覆盖的路径点索引列表
        uncovered_segments: 未覆盖的连续段列表
        validation_time: 验证耗时（秒）
        total_points: 路径点总数
        covered_points: 被覆盖的点数
        uncovered_points: 未被覆盖的点数
        base_radius: 基础覆盖半径（米）
        coverage_reasons: 每个点的覆盖判定原因
    """

    coverage_ratio: float = 0.0
    covered_indices: List[int] = field(default_factory=list)
    uncovered_indices: List[int] = field(default_factory=list)
    uncovered_segments: List[Tuple[int, int]] = field(default_factory=list)
    validation_time: float = 0.0
    total_points: int = 0
    covered_points: int = 0
    uncovered_points: int = 0
    base_radius: float = 0.0
    coverage_reasons: List[str] = field(default_factory=list)

    def is_fully_covered(self) -> bool:
        """检查是否完全覆盖"""
        return self.coverage_ratio == 1.0

    def get_summary(self) -> dict:
        """获取摘要信息"""
        return {
            'coverage_ratio': self.coverage_ratio,
            'total_points': self.total_points,
            'covered_points': self.covered_points,
            'uncovered_points': self.uncovered_points,
            'num_uncovered_segments': len(self.uncovered_segments),
            'validation_time': self.validation_time,
            'base_radius': self.base_radius
        }


class EnhancedCoverageValidator:
    """
    增强版覆盖验证器

    集成覆盖半径计算、障碍物检测、覆盖判定等模块，
    实现基于动态覆盖半径的完整覆盖验证流程。

    Attributes:
        config: IrisZo配置对象
        radius_calculator: 覆盖半径计算器
        obstacle_detector: 障碍物检测器
        coverage_checker: 覆盖判定器
        logger: 日志器

    Example:
        >>> obstacle_map = np.zeros((100, 100), dtype=np.uint8)
        >>> validator = EnhancedCoverageValidator(
        ...     obstacle_map, resolution=0.05, origin=(0.0, 0.0)
        ... )
        >>> result = validator.validate(path, regions)
        >>> print(f"覆盖率: {result.coverage_ratio:.2%}")
    """

    def __init__(
        self,
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float] = (0.0, 0.0),
        config: Optional[IrisZoConfig] = None
    ):
        """
        初始化增强版覆盖验证器

        Args:
            obstacle_map: 障碍物地图（0=自由，1=障碍物）
            resolution: 地图分辨率（米/像素）
            origin: 地图原点（世界坐标）
            config: 配置参数（可选）
        """
        self.config = config or IrisZoConfig()
        self.config.validate()

        # 初始化子模块
        self.radius_calculator = RadiusCalculator(self.config, resolution)
        self.obstacle_detector = ObstacleDetector(
            obstacle_map, resolution, origin, self.config
        )
        self.coverage_checker = CoverageChecker(self.config)

        # 初始化日志
        self.logger = logging.getLogger(__name__)
        log_level = logging.DEBUG if self.config.verbose else logging.INFO
        self.logger.setLevel(log_level)

        # 标记是否已初始化
        self._initialized = False

    def initialize(self) -> None:
        """
        初始化障碍物检测器

        必须在调用validate之前调用此方法。
        """
        if not self._initialized:
            self.obstacle_detector.initialize()
            self._initialized = True
            self.logger.info("EnhancedCoverageValidator initialized")

    def validate(
        self,
        path: np.ndarray,
        regions: List[IrisZoRegion]
    ) -> EnhancedCoverageResult:
        """
        验证路径覆盖情况

        检查路径中的每个点是否被凸区域集合覆盖，使用基于覆盖半径的判定。

        Args:
            path: 路径点数组，shape=(N, dim)
            regions: 凸区域列表

        Returns:
            EnhancedCoverageResult对象

        Example:
            >>> result = validator.validate(path, regions)
            >>> print(f"覆盖率: {result.coverage_ratio:.2%}")
            >>> print(f"基础半径: {result.base_radius:.3f}m")
        """
        start_time = time.time()

        # 确保已初始化
        if not self._initialized:
            self.initialize()

        # 边界情况处理
        if len(path) == 0 or len(regions) == 0:
            return EnhancedCoverageResult(validation_time=time.time() - start_time)

        # 计算基础半径
        base_radius = self.radius_calculator.calculate_base_radius()

        covered_indices = []
        uncovered_indices = []
        coverage_reasons = []

        # 遍历路径点
        for i, point in enumerate(path):
            point_2d = point[:2]  # 取前两维

            is_covered, reason = self._validate_single_point(
                point_2d, base_radius, regions
            )

            coverage_reasons.append(reason)

            if is_covered:
                covered_indices.append(i)
            else:
                uncovered_indices.append(i)

            # 记录DEBUG日志
            self.logger.debug(
                f"Point {i}: {reason}, base_radius={base_radius:.3f}m"
            )

        # 构建结果
        coverage_ratio = len(covered_indices) / len(path)
        uncovered_segments = self._analyze_continuity(uncovered_indices)

        result = EnhancedCoverageResult(
            coverage_ratio=coverage_ratio,
            covered_indices=covered_indices,
            uncovered_indices=uncovered_indices,
            uncovered_segments=uncovered_segments,
            validation_time=time.time() - start_time,
            total_points=len(path),
            covered_points=len(covered_indices),
            uncovered_points=len(uncovered_indices),
            base_radius=base_radius,
            coverage_reasons=coverage_reasons
        )

        # 记录INFO日志
        self.logger.info(
            f"Coverage validation completed: "
            f"ratio={coverage_ratio:.2%}, "
            f"time={result.validation_time:.3f}s, "
            f"base_radius={base_radius:.3f}m"
        )

        return result

    def _validate_single_point(
        self,
        point: np.ndarray,
        base_radius: float,
        regions: List[IrisZoRegion]
    ) -> Tuple[bool, str]:
        """
        验证单个点

        Args:
            point: 点坐标，shape=(2,)
            base_radius: 基础覆盖半径
            regions: 凸区域列表

        Returns:
            (是否覆盖, 判定原因) 元组
        """
        # 检查是否在障碍物内
        if self.obstacle_detector.is_in_obstacle(point):
            return False, "in_obstacle"

        # 查询障碍物距离
        obstacle_distance = self.obstacle_detector.query_distance(point)

        # 调整有效半径
        effective_radius = self.radius_calculator.adjust_effective_radius(
            base_radius, obstacle_distance
        )

        # 判定覆盖
        return self.coverage_checker.check_point_coverage(
            point, effective_radius, regions
        )

    def _analyze_continuity(
        self,
        uncovered_indices: List[int]
    ) -> List[Tuple[int, int]]:
        """
        分析未覆盖段的连续性

        Args:
            uncovered_indices: 未覆盖点索引列表

        Returns:
            未覆盖连续段列表
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

    def check_point_coverage(
        self,
        point: np.ndarray,
        regions: List[IrisZoRegion]
    ) -> bool:
        """
        检查单点覆盖（兼容旧版本API）

        Args:
            point: 点坐标
            regions: 凸区域列表

        Returns:
            True如果被覆盖
        """
        # 确保已初始化
        if not self._initialized:
            self.initialize()

        # 计算基础半径
        base_radius = self.radius_calculator.calculate_base_radius()

        # 验证单点
        point_2d = point[:2]
        is_covered, _ = self._validate_single_point(point_2d, base_radius, regions)

        return is_covered
