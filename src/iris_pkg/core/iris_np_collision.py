"""
IrisNp 碰撞检测模块

包含优化的碰撞检测器和LRU缓存实现。

作者: Path Planning Team
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from .lru_cache import LRUCache


class SimpleCollisionCheckerForIrisNp:
    """优化的碰撞检测器 - 支持缓存和批量检测"""

    def __init__(self, obstacle_map: np.ndarray, resolution: float,
                 origin: Tuple[float, float] = (0.0, 0.0),
                 enable_cache: bool = True, cache_size: int = 10000):
        self.obstacle_map = obstacle_map
        self.resolution = resolution
        self.origin = origin
        self.height, self.width = obstacle_map.shape
        self.enable_cache = enable_cache
        self._cache = LRUCache(cache_size) if enable_cache else None
        self._obstacle_bounds = self._compute_obstacle_bounds()

    def _compute_obstacle_bounds(self) -> Optional[np.ndarray]:
        """计算障碍物边界 - 使用NumPy数组优化"""
        obstacle_indices = np.where(self.obstacle_map > 0)
        if len(obstacle_indices[0]) == 0:
            return None
        # 使用NumPy数组存储边界 [x_min, x_max, y_min, y_max]
        return np.array([
            obstacle_indices[1].min(),  # x_min
            obstacle_indices[1].max(),  # x_max
            obstacle_indices[0].min(),  # y_min
            obstacle_indices[0].max()   # y_max
        ], dtype=np.int32)

    def check_collision(self, point: np.ndarray, safety_margin: float = 0.0) -> bool:
        """
        检查点是否碰撞 - 增强版

        修复漏洞4: 使用更精细的网格检查
        检查点周围的多个网格,避免因精度损失导致的漏检
        """
        x, y = point[0], point[1]
        gx = int((x - self.origin[0]) / self.resolution)
        gy = int((y - self.origin[1]) / self.resolution)

        # 检查缓存
        if self.enable_cache and safety_margin == 0.0 and self._cache:
            cached = self._cache.get((gx, gy))
            if cached is not None:
                return cached

        # 检查边界
        if not (0 <= gx < self.width and 0 <= gy < self.height):
            result = True
        elif safety_margin > 0.0:
            result = self._check_collision_with_margin(gx, gy, int(np.ceil(safety_margin / self.resolution)))
        else:
            # 修复漏洞4: 检查点周围的多个网格
            # 使用3x3的邻域检查,避免因网格边界导致的漏检
            result = self._check_collision_with_neighborhood(gx, gy)

        # 更新缓存
        if self.enable_cache and safety_margin == 0.0 and self._cache:
            self._cache.put((gx, gy), result)

        return result

    def _check_collision_with_neighborhood(self, gx: int, gy: int) -> bool:
        """
        检查点及其邻域的碰撞

        使用3x3邻域检查,确保不会因为网格边界而漏检障碍物

        Args:
            gx: 网格x坐标
            gy: 网格y坐标

        Returns:
            True如果碰撞, False否则
        """
        # 首先检查中心点
        if self.obstacle_map[gy, gx] > 0:
            return True

        # 检查3x3邻域
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # 跳过中心点(已检查)

                check_x = gx + dx
                check_y = gy + dy

                # 检查边界
                if 0 <= check_x < self.width and 0 <= check_y < self.height:
                    if self.obstacle_map[check_y, check_x] > 0:
                        return True

        return False

    def _check_collision_with_margin(self, gx: int, gy: int, margin_cells: int) -> bool:
        if self.obstacle_map[gy, gx] > 0:
            return True
        for dx in range(-margin_cells, margin_cells + 1):
            for dy in range(-margin_cells, margin_cells + 1):
                if dx == 0 and dy == 0:
                    continue
                check_x, check_y = gx + dx, gy + dy
                if 0 <= check_x < self.width and 0 <= check_y < self.height:
                    if self.obstacle_map[check_y, check_x] > 0:
                        return True
        return False

    def get_cache_stats(self) -> Dict[str, Any]:
        if self.enable_cache and self._cache:
            stats = self._cache.get_stats()
            stats['cache_size'] = stats['size']
            return stats
        return {'hits': 0, 'misses': 0, 'hit_rate': 0.0, 'size': 0, 'capacity': 0, 'cache_size': 0}

    def check_collision_batch(self, points: np.ndarray, safety_margin: float = 0.0) -> np.ndarray:
        """
        批量碰撞检测 - 向量化优化版本

        Args:
            points: 点数组 (N x 2)
            safety_margin: 安全裕度（米）

        Returns:
            碰撞结果数组 (N,), True表示碰撞
        """
        n = len(points)
        results = np.zeros(n, dtype=np.bool_)

        # 转换为网格坐标（向量化）
        gx = ((points[:, 0] - self.origin[0]) / self.resolution).astype(np.int32)
        gy = ((points[:, 1] - self.origin[1]) / self.resolution).astype(np.int32)

        # 检查缓存（如果启用）
        if self.enable_cache and safety_margin == 0.0 and self._cache:
            # 检查缓存
            for i in range(n):
                cached = self._cache.get((gx[i], gy[i]))
                if cached is not None:
                    results[i] = cached
                else:
                    # 需要计算
                    results[i] = self._check_single_collision(gx[i], gy[i], safety_margin)
                    self._cache.put((gx[i], gy[i]), results[i])
        else:
            # 批量检查边界
            in_bounds = (0 <= gx) & (gx < self.width) & (0 <= gy) & (gy < self.height)

            # 边界外的点标记为碰撞
            results[~in_bounds] = True

            # 边界内的点检查障碍物
            if self._obstacle_bounds is not None:
                # 使用NumPy数组访问边界
                x_min, x_max, y_min, y_max = self._obstacle_bounds
                # 使用障碍物边界优化
                in_obstacle_bounds = (
                    (x_min <= gx) & (gx <= x_max) &
                    (y_min <= gy) & (gy <= y_max)
                )
                # 在障碍物边界内且在地图边界内的点
                check_mask = in_bounds & in_obstacle_bounds
                valid_indices = np.where(check_mask)[0]

                for idx in valid_indices:
                    results[idx] = self.obstacle_map[gy[idx], gx[idx]] > 0
            else:
                # 没有障碍物边界,检查所有边界内的点
                valid_indices = np.where(in_bounds)[0]
                for idx in valid_indices:
                    results[idx] = self.obstacle_map[gy[idx], gx[idx]] > 0

        return results

    def _check_single_collision(self, gx: int, gy: int, safety_margin: float) -> bool:
        """检查单个点的碰撞"""
        if not (0 <= gx < self.width and 0 <= gy < self.height):
            return True
        elif safety_margin > 0.0:
            return self._check_collision_with_margin(gx, gy, int(np.ceil(safety_margin / self.resolution)))
        elif self._obstacle_bounds is not None:
            # 使用NumPy数组访问边界
            x_min, x_max, y_min, y_max = self._obstacle_bounds
            in_bounds = (x_min <= gx <= x_max) and (y_min <= gy <= y_max)
            return self.obstacle_map[gy, gx] > 0 if in_bounds else False
        else:
            return self.obstacle_map[gy, gx] > 0
