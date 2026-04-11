"""
自定义IrisZo算法碰撞检测适配模块

实现碰撞检测器适配器,将项目的障碍物地图适配到自定义IrisZo算法所需的接口。

作者: Path Planning Team
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from collections import OrderedDict


class LRUCache:
    """
    LRU(Least Recently Used)缓存实现

    使用OrderedDict实现O(1)时间复杂度的get和put操作。

    Attributes:
        cache: OrderedDict存储缓存数据
        capacity: 缓存容量
        hits: 缓存命中次数
        misses: 缓存未命中次数

    Example:
        >>> cache = LRUCache(capacity=1000)
        >>> cache.put((10, 20), True)
        >>> result = cache.get((10, 20))  # 返回True
        >>> stats = cache.get_stats()
    """

    def __init__(self, capacity: int):
        """
        初始化LRU缓存

        Args:
            capacity: 缓存容量
        """
        self.cache = OrderedDict()
        self.capacity = capacity
        self.hits = 0
        self.misses = 0

    def get(self, key: Tuple[int, int]) -> Optional[bool]:
        """
        从缓存中获取值

        Args:
            key: 缓存键,通常为栅格坐标(gx, gy)

        Returns:
            缓存值(True/False),如果不存在则返回None
        """
        if key not in self.cache:
            self.misses += 1
            return None

        # 将访问的键移到末尾(最近使用)
        self.cache.move_to_end(key)
        self.hits += 1
        return self.cache[key]

    def put(self, key: Tuple[int, int], value: bool) -> None:
        """
        向缓存中添加值

        Args:
            key: 缓存键
            value: 缓存值
        """
        if key in self.cache:
            # 如果键已存在,移到末尾
            self.cache.move_to_end(key)
        else:
            # 添加新键值对
            self.cache[key] = value

            # 如果超过容量,移除最久未使用的项
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            包含命中率、命中次数、未命中次数等统计信息的字典
        """
        total = self.hits + self.misses
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / total if total > 0 else 0.0,
            'size': len(self.cache),
            'capacity': self.capacity
        }


class CollisionCheckerAdapter:
    """
    碰撞检测器适配器

    将项目的障碍物地图适配到自定义IrisZo算法所需的碰撞检测接口。
    支持LRU缓存优化和批量检测。

    Attributes:
        obstacle_map: 障碍物地图,0=自由空间,1=障碍物
        resolution: 地图分辨率(米/像素)
        origin: 地图原点坐标(x, y)
        height: 地图高度(像素)
        width: 地图宽度(像素)
        enable_cache: 是否启用缓存
        _cache: LRU缓存实例

    Example:
        >>> import numpy as np
        >>>
        >>> # 创建障碍物地图
        >>> obstacle_map = np.zeros((100, 100), dtype=np.uint8)
        >>> obstacle_map[40:60, 40:60] = 1  # 中心障碍物
        >>>
        >>> # 创建碰撞检测器
        >>> checker = CollisionCheckerAdapter(
        ...     obstacle_map=obstacle_map,
        ...     resolution=0.05,
        ...     origin=(0.0, 0.0)
        ... )
        >>>
        >>> # 检查配置点
        >>> q = np.array([2.5, 2.5])
        >>> is_free = checker.check_config_collision_free(q)
    """

    def __init__(
        self,
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float] = (0.0, 0.0),
        enable_cache: bool = True,
        cache_size: int = 10000,
        safety_margin: float = 0.0
    ):
        """
        初始化碰撞检测器适配器

        Args:
            obstacle_map: 障碍物地图,0=自由空间,1=障碍物
            resolution: 地图分辨率(米/像素)
            origin: 地图原点坐标(x, y)
            enable_cache: 是否启用缓存
            cache_size: 缓存大小
            safety_margin: 安全边界(米),在障碍物周围扩展的安全区域
        """
        self.obstacle_map = obstacle_map
        self.resolution = resolution
        self.origin = origin
        self.height, self.width = obstacle_map.shape
        self.enable_cache = enable_cache
        self.safety_margin = safety_margin

        # 创建LRU缓存
        self._cache = LRUCache(cache_size) if enable_cache else None

        # 计算安全边界的栅格数
        self.safety_margin_grid = int(np.ceil(safety_margin / resolution))

    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        将世界坐标转换为栅格坐标

        Args:
            x: 世界坐标x
            y: 世界坐标y

        Returns:
            栅格坐标(gx, gy)
        """
        gx = int((x - self.origin[0]) / self.resolution)
        gy = int((y - self.origin[1]) / self.resolution)
        return (gx, gy)

    def _check_grid_collision(self, gx: int, gy: int) -> bool:
        """
        检查栅格坐标是否在障碍物内

        Args:
            gx: 栅格x坐标
            gy: 栅格y坐标

        Returns:
            True如果无碰撞,False如果有碰撞
        """
        # 检查边界
        if not (0 <= gx < self.width and 0 <= gy < self.height):
            return False  # 超出边界视为碰撞

        # 检查障碍物
        if self.safety_margin_grid > 0:
            # 使用安全边界:检查周围区域
            for dx in range(-self.safety_margin_grid, self.safety_margin_grid + 1):
                for dy in range(-self.safety_margin_grid, self.safety_margin_grid + 1):
                    check_gx = gx + dx
                    check_gy = gy + dy
                    if 0 <= check_gx < self.width and 0 <= check_gy < self.height:
                        if self.obstacle_map[check_gy, check_gx] == 1:
                            return False
            return True
        else:
            # 不使用安全边界:直接检查
            return (self.obstacle_map[gy, gx] == 0)

    def check_config_collision_free(
        self,
        q: np.ndarray,
        context_number: Optional[int] = None
    ) -> bool:
        """
        检查配置点是否无碰撞

        Args:
            q: 配置点坐标,shape=(dim,),通常为[x, y]或[x, y, theta]
            context_number: 上下文编号(并行时使用),当前未使用

        Returns:
            True如果无碰撞,False如果有碰撞

        Example:
            >>> q = np.array([2.5, 2.5])
            >>> is_free = checker.check_config_collision_free(q)
        """
        # 提取x, y坐标
        x, y = q[0], q[1]

        # 转换为栅格坐标
        gx, gy = self._world_to_grid(x, y)

        # 检查缓存
        if self.enable_cache and self._cache is not None:
            cached = self._cache.get((gx, gy))
            if cached is not None:
                return cached

        # 执行碰撞检测
        result = self._check_grid_collision(gx, gy)

        # 更新缓存
        if self.enable_cache and self._cache is not None:
            self._cache.put((gx, gy), result)

        return result

    def check_edge_collision_free(
        self,
        q1: np.ndarray,
        q2: np.ndarray,
        context_number: Optional[int] = None
    ) -> bool:
        """
        检查连接两个配置点的边是否无碰撞

        沿边采样多个点进行检查。

        Args:
            q1: 起点配置,shape=(dim,)
            q2: 终点配置,shape=(dim,)
            context_number: 上下文编号,当前未使用

        Returns:
            True如果整条边无碰撞,False否则

        Example:
            >>> q1 = np.array([0.0, 0.0])
            >>> q2 = np.array([5.0, 5.0])
            >>> is_free = checker.check_edge_collision_free(q1, q2)
        """
        # 计算边的长度
        distance = np.linalg.norm(q2[:2] - q1[:2])

        # 根据分辨率确定采样步数
        # 确保采样间隔不超过分辨率
        num_samples = max(int(distance / self.resolution) + 1, 2)

        # 沿边采样检查
        for i in range(num_samples + 1):
            t = i / num_samples
            q = q1 + t * (q2 - q1)
            if not self.check_config_collision_free(q):
                return False

        return True

    def check_configs_collision_free(
        self,
        configs: List[np.ndarray],
        parallelize: bool = True
    ) -> List[bool]:
        """
        批量检查多个配置点是否无碰撞

        Args:
            configs: 配置点列表
            parallelize: 是否并行处理(当前未实现并行)

        Returns:
            布尔结果列表

        Example:
            >>> configs = [np.array([1.0, 1.0]), np.array([2.0, 2.0])]
            >>> results = checker.check_configs_collision_free(configs)
        """
        # TODO: 实现并行处理
        # 当前使用串行处理
        return [self.check_config_collision_free(q) for q in configs]

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            缓存统计信息字典
        """
        if self.enable_cache and self._cache is not None:
            return self._cache.get_stats()
        else:
            return {
                'hits': 0,
                'misses': 0,
                'hit_rate': 0.0,
                'size': 0,
                'capacity': 0
            }

    def clear_cache(self) -> None:
        """清空缓存"""
        if self.enable_cache and self._cache is not None:
            self._cache.clear()

    def get_map_info(self) -> Dict[str, Any]:
        """
        获取地图信息

        Returns:
            地图信息字典
        """
        return {
            'height': self.height,
            'width': self.width,
            'resolution': self.resolution,
            'origin': self.origin,
            'safety_margin': self.safety_margin,
            'obstacle_count': int(np.sum(self.obstacle_map == 1)),
            'free_count': int(np.sum(self.obstacle_map == 0))
        }

    def __str__(self) -> str:
        """
        返回碰撞检测器的字符串表示

        Returns:
            格式化的字符串
        """
        map_info = self.get_map_info()
        cache_stats = self.get_cache_stats()
        return (
            f"CollisionCheckerAdapter(\n"
            f"  地图尺寸: {map_info['width']}x{map_info['height']}\n"
            f"  分辨率: {map_info['resolution']}\n"
            f"  原点: {map_info['origin']}\n"
            f"  障碍物数量: {map_info['obstacle_count']}\n"
            f"  缓存命中率: {cache_stats['hit_rate']:.2%}\n"
            f")"
        )
