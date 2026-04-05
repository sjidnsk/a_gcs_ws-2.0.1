"""
IrisNp 路径覆盖验证模块

负责验证路径是否被凸区域完全覆盖，并处理未覆盖的路径点。

作者: Path Planning Team
"""

import numpy as np
from typing import List, Tuple, Optional
from pydrake.geometry.optimization import HPolyhedron

from ..config.iris_np_config import IrisNpConfig
from .iris_np_region_data import IrisNpRegion
from .iris_np_collision import SimpleCollisionCheckerForIrisNp
from .iris_np_expansion import IrisNpExpansion


class IrisNpCoverageChecker:
    """IrisNp 路径覆盖验证器"""

    def __init__(
        self,
        config: IrisNpConfig,
        expansion: IrisNpExpansion
    ):
        """
        初始化覆盖验证器

        Args:
            config: IrisNp 配置参数
            expansion: IrisNp 膨胀器
        """
        self.config = config
        self.expansion = expansion

    def verify_path_coverage(
        self,
        path: List[Tuple[float, float, float]],
        regions: List[IrisNpRegion]
    ) -> bool:
        """
        验证路径是否完全被凸区域覆盖

        Args:
            path: 路径点列表
            regions: 凸区域列表

        Returns:
            True 如果路径完全被覆盖，False 否则
        """
        if len(path) == 0:
            return True

        if len(regions) == 0:
            return False

        # 检查每个路径点
        uncovered_points = []
        for i, (x, y, _) in enumerate(path):
            point = np.array([x, y])
            is_covered = False

            # 检查点是否在任何区域内
            for region in regions:
                if region.contains(point, tol=1e-6):
                    is_covered = True
                    break

            if not is_covered:
                uncovered_points.append(i)

        if len(uncovered_points) > 0:
            if self.config.verbose:
                print(f"  警告: 发现 {len(uncovered_points)} 个路径点未被覆盖")
                print(f"  未覆盖点索引: {uncovered_points[:10]}{'...' if len(uncovered_points) > 10 else ''}")
            return False

        return True

    def find_uncovered_points(
        self,
        path: List[Tuple[float, float, float]],
        regions: List[IrisNpRegion]
    ) -> List[int]:
        """
        方案2: 查找未覆盖的路径点索引

        Args:
            path: 路径点列表
            regions: 凸区域列表

        Returns:
            未覆盖点的索引列表
        """
        if len(path) == 0 or len(regions) == 0:
            return list(range(len(path)))

        uncovered_indices = []
        for i, (x, y, _) in enumerate(path):
            point = np.array([x, y])
            is_covered = False

            # 检查点是否在任何区域内
            for region in regions:
                if region.contains(point, tol=1e-6):
                    is_covered = True
                    break

            if not is_covered:
                uncovered_indices.append(i)

        return uncovered_indices

    def generate_regions_for_uncovered_points(
        self,
        path: List[Tuple[float, float, float]],
        uncovered_indices: List[int],
        checker: SimpleCollisionCheckerForIrisNp,
        domain: HPolyhedron,
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float]
    ) -> List[IrisNpRegion]:
        """
        方案2: 为未覆盖点生成小区域

        使用更小的初始区域和更密集的采样,确保覆盖

        Args:
            path: 路径点列表
            uncovered_indices: 未覆盖点索引列表
            checker: 碰撞检测器
            domain: 定义域
            obstacle_map: 障碍物地图
            resolution: 分辨率
            origin: 原点

        Returns:
            生成的凸区域列表
        """
        regions = []

        # 对未覆盖点进行聚类,避免生成过多重叠区域
        # 使用简单的距离聚类
        clusters = self._cluster_uncovered_points(path, uncovered_indices, cluster_distance=2.0)

        if self.config.verbose:
            print(f"  未覆盖点聚类为 {len(clusters)} 个簇")

        # 为每个簇生成多个区域（使用簇中的多个点作为种子点）
        for cluster_idx, cluster_indices in enumerate(clusters):
            # 策略：使用簇中的多个点作为种子点，而不是只用中心点
            # 优先使用簇的中心点，然后尝试其他点
            seed_candidates = []
            
            # 1. 添加中心点
            cluster_points = [path[i] for i in cluster_indices]
            center_x = sum(p[0] for p in cluster_points) / len(cluster_points)
            center_y = sum(p[1] for p in cluster_points) / len(cluster_points)
            seed_candidates.append(np.array([center_x, center_y]))
            
            # 2. 添加簇中的其他点（每隔2个点取一个，避免过多）
            for i in range(0, len(cluster_indices), max(1, len(cluster_indices) // 3)):
                idx = cluster_indices[i]
                x, y, _ = path[idx]
                seed_candidates.append(np.array([x, y]))
            
            # 尝试每个候选种子点
            region_generated = False
            for seed_idx, seed_point in enumerate(seed_candidates):
                # 检查种子点是否在障碍物内
                gx = int((seed_point[0] - origin[0]) / resolution)
                gy = int((seed_point[1] - origin[1]) / resolution)

                if not (0 <= gx < obstacle_map.shape[1] and 0 <= gy < obstacle_map.shape[0]):
                    continue

                if obstacle_map[gy, gx] > 0:
                    continue

                # 使用更小的初始区域生成凸区域
                try:
                    # 临时修改配置,使用更小的初始区域
                    original_initial_size = self.config.initial_region_size
                    original_max_size = self.config.max_region_size

                    # 使用更小的初始区域(0.05米)和更小的最大区域(20米)
                    self.config.initial_region_size = 0.05
                    self.config.max_region_size = 20.0

                    region = self.expansion.simplified_iris_with_sampling(
                        checker, seed_point, domain, obstacle_map, resolution, origin
                    )

                    # 恢复原始配置
                    self.config.initial_region_size = original_initial_size
                    self.config.max_region_size = original_max_size

                    if region is not None and region.area > 0:
                        regions.append(region)
                        region_generated = True
                        if self.config.verbose:
                            print(f"    ✓ 簇 {cluster_idx + 1} 种子点 {seed_idx + 1} 生成区域成功，面积: {region.area:.2f} 平方米")
                        break  # 成功生成一个区域后，尝试下一个簇

                except Exception as e:
                    if self.config.verbose:
                        print(f"    ✗ 簇 {cluster_idx + 1} 种子点 {seed_idx + 1} 生成区域失败: {e}")
                    continue
            
            if not region_generated and self.config.verbose:
                print(f"    ✗ 簇 {cluster_idx + 1} 所有种子点都未能生成有效区域")

        return regions

    def _cluster_uncovered_points(
        self,
        path: List[Tuple[float, float, float]],
        uncovered_indices: List[int],
        cluster_distance: float = 2.0
    ) -> List[List[int]]:
        """
        对未覆盖点进行聚类

        Args:
            path: 路径点列表
            uncovered_indices: 未覆盖点索引列表
            cluster_distance: 聚类距离阈值

        Returns:
            聚类结果,每个元素是一个簇的索引列表
        """
        if len(uncovered_indices) == 0:
            return []

        # 简单的贪心聚类算法
        clusters = []
        remaining = set(uncovered_indices)

        while remaining:
            # 选择一个种子点
            seed_idx = remaining.pop()
            cluster = [seed_idx]

            # 扩展簇
            changed = True
            while changed:
                changed = False
                for idx in list(remaining):
                    # 检查是否与簇中任何点距离小于阈值
                    for cluster_idx in cluster:
                        x1, y1, _ = path[idx]
                        x2, y2, _ = path[cluster_idx]
                        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        if dist <= cluster_distance:
                            cluster.append(idx)
                            remaining.remove(idx)
                            changed = True
                            break

            clusters.append(cluster)

        return clusters
