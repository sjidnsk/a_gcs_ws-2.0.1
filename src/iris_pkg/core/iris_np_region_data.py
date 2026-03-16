"""
IrisNp 区域数据模块

包含 IrisNp 生成的凸区域数据结构和区域索引。

作者: Path Planning Team
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from scipy.spatial import KDTree


class RegionIndex:
    """区域空间索引 - 使用KDTree优化区域查询"""

    def __init__(self, regions: List['IrisNpRegion']):
        """
        初始化区域索引

        Args:
            regions: 区域列表
        """
        self.regions = regions
        self.num_regions = len(regions)

        if self.num_regions > 0:
            # 构建区域中心的KDTree
            self.centroids = np.array([r.centroid for r in regions])
            self.kdtree = KDTree(self.centroids)

            # 预计算每个区域的边界框
            self.bboxes = []
            for region in regions:
                x_min, y_min = region.vertices.min(axis=0)
                x_max, y_max = region.vertices.max(axis=0)
                self.bboxes.append((x_min, y_min, x_max, y_max))
        else:
            self.centroids = None
            self.kdtree = None
            self.bboxes = []

    def find_nearby_regions(self, point: np.ndarray, radius: float) -> List[int]:
        """
        查找附近的区域索引

        Args:
            point: 查询点
            radius: 搜索半径

        Returns:
            附近区域的索引列表
        """
        if self.kdtree is None:
            return []

        # 使用KDTree查询半径内的区域中心
        nearby_indices = self.kdtree.query_ball_point(point, radius)
        return list(nearby_indices)

    def find_containing_regions(self, point: np.ndarray, tol: float = 1e-6) -> List[int]:
        """
        查找包含指定点的区域索引

        Args:
            point: 查询点
            tol: 容差

        Returns:
            包含该点的区域索引列表
        """
        if self.num_regions == 0:
            return []

        # 计算到所有区域中心的距离
        distances = np.linalg.norm(self.centroids - point, axis=1)

        # 找到最大区域半径作为搜索半径
        max_radius = max(
            np.max(np.linalg.norm(r.vertices - r.centroid, axis=1))
            for r in self.regions
        )

        # 只检查距离在最大半径内的区域
        candidate_indices = np.where(distances <= max_radius * 1.5)[0]

        # 检查这些区域是否包含该点
        containing_indices = []
        for idx in candidate_indices:
            if self.regions[idx].contains(point, tol):
                containing_indices.append(idx)

        return containing_indices

    def is_point_covered(self, point: np.ndarray, tol: float = 1e-6) -> bool:
        """
        检查点是否被任何区域覆盖

        Args:
            point: 查询点
            tol: 容差

        Returns:
            True如果点被覆盖
        """
        return len(self.find_containing_regions(point, tol)) > 0


@dataclass
class IrisNpRegion:
    """IrisNp 生成的凸区域"""
    # 多面体表示 (Ax <= b)
    A: np.ndarray  # 约束矩阵
    b: np.ndarray  # 约束向量

    # 几何信息
    vertices: np.ndarray  # 顶点坐标 (N x 2)
    centroid: np.ndarray  # 实际几何中心点
    area: float  # 面积

    # 元数据
    seed_point: np.ndarray  # 原始种子点（从路径提取的点）
    expansion_center: np.ndarray = None  # 膨胀中心（用于生成区域的中心）
    region_id: int = 0

    def contains(self, point: np.ndarray, tol: float = 1e-6) -> bool:
        """检查点是否在区域内"""
        return np.all(self.A @ point <= self.b + tol)

    def get_vertices_ordered(self) -> np.ndarray:
        """获取按角度排序的顶点（用于绘图）"""
        if len(self.vertices) == 0:
            return self.vertices

        # 计算相对于中心的角度
        angles = np.arctan2(
            self.vertices[:, 1] - self.centroid[1],
            self.vertices[:, 0] - self.centroid[0]
        )
        # 按角度排序
        sorted_indices = np.argsort(angles)
        return self.vertices[sorted_indices]


@dataclass
class IrisNpResult:
    """IrisNp 处理结果"""
    # 生成的凸区域列表
    regions: List[IrisNpRegion] = field(default_factory=list)

    # 统计信息
    num_regions: int = 0
    total_area: float = 0.0
    coverage_ratio: float = 0.0

    # 处理时间
    iris_time: float = 0.0
    postprocess_time: float = 0.0
    total_time: float = 0.0

    # 配置
    config: Any = None
