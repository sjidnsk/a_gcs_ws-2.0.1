"""
障碍物检测模块

封装距离查询引擎，提供统一的障碍物检测接口。

作者: Path Planning Team
"""

from typing import Tuple, Optional, TYPE_CHECKING
import numpy as np

from .iriszo_distance_query import (
    DistanceQueryEngine,
    DistanceTransformEngine,
    KDTreeEngine
)

if TYPE_CHECKING:
    from ..config.iriszo_config import IrisZoConfig


class ObstacleDetector:
    """
    障碍物检测器

    封装距离查询引擎，提供统一的障碍物检测接口，包括距离查询、
    障碍物内检测、地图范围检测等功能。

    Attributes:
        obstacle_map: 障碍物地图
        resolution: 地图分辨率
        origin: 地图原点
        config: 配置参数
        distance_engine: 距离查询引擎
        height: 地图高度
        width: 地图宽度

    Example:
        >>> obstacle_map = np.zeros((100, 100), dtype=np.uint8)
        >>> obstacle_map[40:60, 40:60] = 1
        >>> detector = ObstacleDetector(obstacle_map, 0.05, (0.0, 0.0))
        >>> detector.initialize()
        >>> distance = detector.query_distance(np.array([2.5, 2.5]))
        >>> is_in_obs = detector.is_in_obstacle(np.array([2.5, 2.5]))
    """

    def __init__(
        self,
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float] = (0.0, 0.0),
        config: Optional["IrisZoConfig"] = None
    ):
        """
        初始化障碍物检测器

        Args:
            obstacle_map: 障碍物地图（0=自由，1=障碍物）
            resolution: 地图分辨率（米/像素）
            origin: 地图原点（世界坐标）
            config: 配置参数（可选）

        Raises:
            ValueError: 如果参数无效
        """
        if obstacle_map.ndim != 2:
            raise ValueError(
                f"obstacle_map必须是2D数组，当前维度: {obstacle_map.ndim}"
            )

        if resolution <= 0:
            raise ValueError(f"resolution必须>0，当前值: {resolution}")

        self.obstacle_map = obstacle_map
        self.resolution = resolution
        self.origin = origin

        # 延迟导入避免循环依赖
        if config is None:
            from ..config.iriszo_config import IrisZoConfig
            config = IrisZoConfig()

        self.config = config

        self.distance_engine: Optional[DistanceQueryEngine] = None
        self.height, self.width = obstacle_map.shape

    def initialize(self) -> None:
        """
        初始化距离查询引擎

        根据配置选择距离变换或KD-Tree引擎。
        """
        if self.config.enable_distance_transform:
            self.distance_engine = DistanceTransformEngine(
                self.obstacle_map,
                self.resolution,
                self.origin
            )
        else:
            self.distance_engine = KDTreeEngine(
                self.obstacle_map,
                self.resolution,
                self.origin
            )

        self.distance_engine.initialize()

    def query_distance(self, point: np.ndarray) -> float:
        """
        查询到最近障碍物的距离

        Args:
            point: 查询点坐标（世界坐标系），shape=(2,)

        Returns:
            到最近障碍物的距离（米）

        Raises:
            RuntimeError: 如果检测器未初始化
        """
        if self.distance_engine is None or not self.distance_engine.is_initialized():
            raise RuntimeError("障碍物检测器未初始化，请先调用initialize()")

        return self.distance_engine.query(point)

    def is_in_obstacle(self, point: np.ndarray) -> bool:
        """
        检查点是否在障碍物内

        Args:
            point: 查询点坐标（世界坐标系），shape=(2,)

        Returns:
            True如果点在障碍物内，False否则
        """
        # 转换为栅格坐标
        grid_x = int((point[0] - self.origin[0]) / self.resolution)
        grid_y = int((point[1] - self.origin[1]) / self.resolution)

        # 边界检查
        if not (0 <= grid_x < self.width and 0 <= grid_y < self.height):
            return False  # 地图外视为自由空间

        return self.obstacle_map[grid_y, grid_x] == 1

    def is_in_map(self, point: np.ndarray) -> bool:
        """
        检查点是否在地图范围内

        Args:
            point: 查询点坐标（世界坐标系），shape=(2,)

        Returns:
            True如果点在地图内，False否则
        """
        grid_x = (point[0] - self.origin[0]) / self.resolution
        grid_y = (point[1] - self.origin[1]) / self.resolution

        return 0 <= grid_x < self.width and 0 <= grid_y < self.height

    def get_map_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        获取地图边界

        Returns:
            ((x_min, x_max), (y_min, y_max)) 地图边界坐标
        """
        x_min = self.origin[0]
        x_max = self.origin[0] + self.width * self.resolution
        y_min = self.origin[1]
        y_max = self.origin[1] + self.height * self.resolution

        return ((x_min, x_max), (y_min, y_max))
