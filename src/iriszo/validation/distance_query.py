"""
障碍物距离查询引擎

实现高效的障碍物距离查询，支持距离变换和KD-Tree两种方式。

作者: Path Planning Team
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from scipy.spatial import KDTree
    KDTree_AVAILABLE = True
except ImportError:
    KDTree_AVAILABLE = False


class DistanceQueryEngine(ABC):
    """
    距离查询引擎抽象基类

    定义距离查询的统一接口。
    """

    @abstractmethod
    def query(self, point: np.ndarray) -> float:
        """
        查询到最近障碍物的距离

        Args:
            point: 查询点坐标（世界坐标系），shape=(2,)

        Returns:
            到最近障碍物的距离（米）
        """
        pass

    @abstractmethod
    def is_initialized(self) -> bool:
        """
        检查引擎是否已初始化

        Returns:
            True如果已初始化，False否则
        """
        pass


class DistanceTransformEngine(DistanceQueryEngine):
    """
    基于距离变换的查询引擎

    使用OpenCV的距离变换预计算距离场，实现O(1)查询。

    Attributes:
        obstacle_map: 障碍物地图
        resolution: 地图分辨率
        origin: 地图原点
        distance_field: 预计算的距离场

    Example:
        >>> obstacle_map = np.zeros((100, 100), dtype=np.uint8)
        >>> obstacle_map[40:60, 40:60] = 1
        >>> engine = DistanceTransformEngine(obstacle_map, 0.05, (0.0, 0.0))
        >>> engine.initialize()
        >>> distance = engine.query(np.array([2.5, 2.5]))
    """

    def __init__(
        self,
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float]
    ):
        """
        初始化距离变换引擎

        Args:
            obstacle_map: 障碍物地图（0=自由，1=障碍物）
            resolution: 地图分辨率（米/像素）
            origin: 地图原点（世界坐标）
        """
        self.obstacle_map = obstacle_map
        self.resolution = resolution
        self.origin = origin
        self.distance_field = None

    def initialize(self) -> None:
        """
        预计算距离场

        使用OpenCV的距离变换计算每个点到最近障碍物的距离。

        Raises:
            RuntimeError: 如果OpenCV不可用
        """
        if not CV2_AVAILABLE:
            raise RuntimeError(
                "OpenCV不可用，请安装opencv-python或使用KDTreeEngine"
            )

        # 使用OpenCV距离变换
        # 注意：OpenCV的距离变换需要自由空间为非零值
        free_space = (1 - self.obstacle_map).astype(np.uint8)
        self.distance_field = cv2.distanceTransform(
            free_space,
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE
        ) * self.resolution

    def query(self, point: np.ndarray) -> float:
        """
        查询到最近障碍物的距离

        Args:
            point: 查询点坐标（世界坐标系）

        Returns:
            到最近障碍物的距离（米）

        Raises:
            RuntimeError: 如果距离场未初始化
        """
        if self.distance_field is None:
            raise RuntimeError("距离场未初始化，请先调用initialize()")

        # 转换为栅格坐标
        grid_x = int((point[0] - self.origin[0]) / self.resolution)
        grid_y = int((point[1] - self.origin[1]) / self.resolution)

        # 边界检查
        height, width = self.obstacle_map.shape
        if not (0 <= grid_x < width and 0 <= grid_y < height):
            return float('inf')  # 地图外无障碍物

        return float(self.distance_field[grid_y, grid_x])

    def is_initialized(self) -> bool:
        """检查引擎是否已初始化"""
        return self.distance_field is not None


class KDTreeEngine(DistanceQueryEngine):
    """
    基于KD-Tree的查询引擎

    使用SciPy的KD-Tree构建障碍物点索引，实现O(log N)查询。

    Attributes:
        obstacle_map: 障碍物地图
        resolution: 地图分辨率
        origin: 地图原点
        tree: KD-Tree索引
        obstacle_points: 障碍物点坐标

    Example:
        >>> obstacle_map = np.zeros((100, 100), dtype=np.uint8)
        >>> obstacle_map[40:60, 40:60] = 1
        >>> engine = KDTreeEngine(obstacle_map, 0.05, (0.0, 0.0))
        >>> engine.initialize()
        >>> distance = engine.query(np.array([2.5, 2.5]))
    """

    def __init__(
        self,
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float]
    ):
        """
        初始化KD-Tree引擎

        Args:
            obstacle_map: 障碍物地图（0=自由，1=障碍物）
            resolution: 地图分辨率（米/像素）
            origin: 地图原点（世界坐标）
        """
        self.obstacle_map = obstacle_map
        self.resolution = resolution
        self.origin = origin
        self.tree = None
        self.obstacle_points = None

    def initialize(self) -> None:
        """
        构建KD-Tree索引

        提取障碍物点坐标并构建KD-Tree。

        Raises:
            RuntimeError: 如果SciPy不可用
        """
        if not KDTree_AVAILABLE:
            raise RuntimeError(
                "SciPy不可用，请安装scipy或使用DistanceTransformEngine"
            )

        # 提取障碍物点坐标
        obstacle_indices = np.argwhere(self.obstacle_map == 1)

        if len(obstacle_indices) == 0:
            # 无障碍物
            self.obstacle_points = np.array([]).reshape(0, 2)
            self.tree = None
            return

        # 转换为世界坐标
        # obstacle_indices的格式是(row, col)，对应(y, x)
        self.obstacle_points = np.array([
            [
                self.origin[0] + idx[1] * self.resolution,  # x坐标
                self.origin[1] + idx[0] * self.resolution   # y坐标
            ]
            for idx in obstacle_indices
        ])

        # 构建KD-Tree
        self.tree = KDTree(self.obstacle_points)

    def query(self, point: np.ndarray) -> float:
        """
        查询到最近障碍物的距离

        Args:
            point: 查询点坐标（世界坐标系）

        Returns:
            到最近障碍物的距离（米）
        """
        if self.tree is None:
            return float('inf')  # 无障碍物

        distance, _ = self.tree.query(point)
        return float(distance)

    def is_initialized(self) -> bool:
        """检查引擎是否已初始化"""
        return self.tree is not None or (
            self.obstacle_points is not None and len(self.obstacle_points) == 0
        )
