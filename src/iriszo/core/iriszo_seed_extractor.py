"""
自定义IrisZo算法种子点提取模块

实现多种种子点提取策略,包括均匀采样和未覆盖点采样。

作者: Path Planning Team
"""

import numpy as np
from typing import List, Tuple, Optional
import warnings

from ..config.iriszo_config import IrisZoConfig
from .iriszo_region_data import IrisZoRegion


class IrisZoSeedExtractor:
    """
    种子点提取器

    提供多种种子点提取策略:
    - 均匀采样: 沿路径均匀采样
    - 未覆盖点采样: 在未覆盖区域提取种子点

    Attributes:
        config: 配置参数

    Example:
        >>> extractor = IrisZoSeedExtractor(config)
        >>> seeds = extractor.extract_seed_points(
        ...     path, obstacle_map, resolution, origin
        ... )
    """

    def __init__(self, config: IrisZoConfig):
        """
        初始化种子点提取器

        Args:
            config: 配置参数
        """
        self.config = config

    def extract_seed_points(
        self,
        path: List[Tuple[float, float, float]],
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float],
        batch: int = 1,
        existing_regions: Optional[List[IrisZoRegion]] = None
    ) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        提取种子点

        Args:
            path: 路径点列表,每个元素为(x, y, theta)
            obstacle_map: 障碍物地图
            resolution: 地图分辨率
            origin: 地图原点
            batch: 批次编号(1=第一批, 2=第二批)
            existing_regions: 已生成的区域(用于第二批)

        Returns:
            种子点列表,每个元素为(seed_point, tangent_direction)
            tangent_direction为None表示使用各向同性膨胀

        Example:
            >>> seeds = extractor.extract_seed_points(
            ...     path, obstacle_map, 0.05, (0.0, 0.0)
            ... )
        """
        if batch == 1:
            return self._extract_uniform(path, obstacle_map, resolution, origin)
        elif batch == 2:
            return self._extract_uncovered(
                path, obstacle_map, resolution, origin, existing_regions
            )
        else:
            return []

    def _extract_uniform(
        self,
        path: List[Tuple[float, float, float]],
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float]
    ) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        均匀采样策略

        Args:
            path: 路径点列表
            obstacle_map: 障碍物地图
            resolution: 地图分辨率
            origin: 地图原点

        Returns:
            种子点列表
        """
        seeds = []
        existing_points = []  # 仅存seed_point的numpy数组，用于向量化距离检查
        min_distance = 0.5  # 最小种子点距离
        sample_interval = 10  # 采样间隔

        for i in range(0, len(path), sample_interval):
            x, y, _ = path[i]
            seed_point = np.array([x, y])

            # 检查是否在自由空间
            gx = int((x - origin[0]) / resolution)
            gy = int((y - origin[1]) / resolution)

            if 0 <= gx < obstacle_map.shape[1] and 0 <= gy < obstacle_map.shape[0]:
                if obstacle_map[gy, gx] == 0:
                    # 向量化距离检查
                    if len(existing_points) == 0:
                        seeds.append((seed_point, None))
                        existing_points.append(seed_point)
                    else:
                        existing = np.array(existing_points)
                        distances = np.linalg.norm(existing - seed_point, axis=1)
                        if np.all(distances >= min_distance):
                            seeds.append((seed_point, None))
                            existing_points.append(seed_point)

        # 确保包含起点和终点
        if len(path) > 0:
            # 起点
            x0, y0, _ = path[0]
            seed0 = np.array([x0, y0])
            if len(existing_points) == 0:
                seeds.insert(0, (seed0, None))
                existing_points.insert(0, seed0)
            else:
                existing = np.array(existing_points)
                distances = np.linalg.norm(existing - seed0, axis=1)
                if np.all(distances >= min_distance * 0.5):
                    seeds.insert(0, (seed0, None))
                    existing_points.insert(0, seed0)

            # 终点
            x1, y1, _ = path[-1]
            seed1 = np.array([x1, y1])
            existing = np.array(existing_points)
            distances = np.linalg.norm(existing - seed1, axis=1)
            if np.all(distances >= min_distance * 0.5):
                seeds.append((seed1, None))
                existing_points.append(seed1)

        return seeds

    def _extract_uncovered(
        self,
        path: List[Tuple[float, float, float]],
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float],
        existing_regions: Optional[List[IrisZoRegion]]
    ) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        未覆盖点采样策略

        Args:
            path: 路径点列表
            obstacle_map: 障碍物地图
            resolution: 地图分辨率
            origin: 地图原点
            existing_regions: 已生成的区域

        Returns:
            种子点列表
        """
        if existing_regions is None or len(existing_regions) == 0:
            return []

        seeds = []
        existing_points = []  # 仅存seed_point的numpy数组，用于向量化距离检查
        min_distance = 0.3

        # 检查每个路径点是否被覆盖
        for i, (x, y, _) in enumerate(path):
            point = np.array([x, y])

            # 检查是否被任一区域覆盖
            covered = any(region.contains(point) for region in existing_regions)

            if not covered:
                # 未覆盖点,作为候选种子点
                # 计算切线方向
                if i > 0 and i < len(path) - 1:
                    prev_point = np.array([path[i-1][0], path[i-1][1]])
                    next_point = np.array([path[i+1][0], path[i+1][1]])
                    tangent = next_point - prev_point
                    norm = np.linalg.norm(tangent)
                    if norm > 1e-6:
                        tangent = tangent / norm
                    else:
                        tangent = None
                else:
                    tangent = None

                # 向量化距离检查
                if len(existing_points) == 0:
                    seeds.append((point, tangent))
                    existing_points.append(point)
                else:
                    existing = np.array(existing_points)
                    distances = np.linalg.norm(existing - point, axis=1)
                    if np.all(distances >= min_distance):
                        seeds.append((point, tangent))
                        existing_points.append(point)

        return seeds

    def __str__(self) -> str:
        """
        返回提取器的字符串表示

        Returns:
            格式化的字符串
        """
        return "IrisZoSeedExtractor()"
