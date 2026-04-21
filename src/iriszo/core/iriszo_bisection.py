"""
自定义IrisZo算法二分搜索模块

实现二分搜索边界定位器,在每个搜索方向上精确定位碰撞边界。
支持精度自适应和并行化优化。

作者: Path Planning Team
"""

import numpy as np
from typing import List, Tuple
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from pydrake.geometry.optimization import Hyperellipsoid
    DRAKE_AVAILABLE = True
except ImportError:
    DRAKE_AVAILABLE = False
    Hyperellipsoid = None

from ..config.iriszo_config import IrisZoConfig
from .iriszo_collision import CollisionCheckerAdapter


class BisectionSearcher:
    """
    二分搜索边界定位器

    在每个搜索方向上使用二分法精确定位碰撞边界。

    二分搜索算法原理:
        1. 给定起点c(椭球体中心)和碰撞点p
        2. 初始化区间[a, b],其中a=c, b=p
        3. 重复bisection_steps次:
           a. 计算中点m = (a + b) / 2
           b. 如果m有碰撞,则b = m
           c. 否则a = m
        4. 返回b作为边界点

    优化特性:
        - 精度自适应: 根据碰撞点到中心的距离动态调整步数
        - 并行化: 多个碰撞点的二分搜索使用ThreadPoolExecutor并行执行

    Attributes:
        config: 配置参数

    Example:
        >>> searcher = BisectionSearcher(config)
        >>> boundary_points = searcher.search_boundary(
        ...     collision_points, ellipsoid_center, checker
        ... )
    """

    def __init__(self, config: IrisZoConfig):
        """
        初始化二分搜索器

        Args:
            config: 配置参数
        """
        self.config = config

    def search_boundary(
        self,
        collision_points: np.ndarray,
        ellipsoid_center: np.ndarray,
        checker: CollisionCheckerAdapter
    ) -> np.ndarray:
        """
        对多个碰撞点执行二分搜索

        Args:
            collision_points: 碰撞点数组,shape=(N, dim)
            ellipsoid_center: 椭球体中心
            checker: 碰撞检测器

        Returns:
            边界点数组,shape=(N, dim)

        Example:
            >>> boundary_points = searcher.search_boundary(
            ...     collision_points, center, checker
            ... )
        """
        num_points = collision_points.shape[0]
        boundary_points = np.zeros_like(collision_points)

        # 计算自适应步数
        steps_array = self._compute_adaptive_steps(collision_points, ellipsoid_center)

        # 判断是否启用并行
        use_parallel = (
            self.config.enable_parallel_bisection
            and num_points >= self.config.parallel_bisection_threshold
        )

        if use_parallel:
            workers = min(self.config.parallel_bisection_workers, num_points)
            try:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {}
                    for i in range(num_points):
                        future = executor.submit(
                            self._bisection_single,
                            collision_points[i], ellipsoid_center, checker, steps_array[i]
                        )
                        futures[future] = i

                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            boundary_points[idx] = future.result()
                        except Exception as e:
                            warnings.warn(
                                f"碰撞点{idx}的二分搜索失败: {e}, 使用碰撞点本身作为边界点"
                            )
                            boundary_points[idx] = collision_points[idx]
            except Exception as e:
                warnings.warn(f"线程池创建失败: {e}, 回退到串行执行")
                for i in range(num_points):
                    boundary_points[i] = self._bisection_single(
                        collision_points[i], ellipsoid_center, checker, steps_array[i]
                    )
        else:
            # 串行路径
            for i in range(num_points):
                boundary_points[i] = self._bisection_single(
                    collision_points[i], ellipsoid_center, checker, steps_array[i]
                )

        return boundary_points

    def _compute_adaptive_steps(
        self,
        collision_points: np.ndarray,
        center: np.ndarray
    ) -> np.ndarray:
        """
        计算每个碰撞点的自适应二分搜索步数

        距离中心越远的碰撞点,使用越多的步数(远距离区间大,需要更多步数保证精度);
        距离中心越近的碰撞点,使用越少的步数(近距离区间小,少量步数即可满足精度)。

        Args:
            collision_points: 碰撞点数组,shape=(N, dim)
            center: 椭球体中心

        Returns:
            步数数组,shape=(N,)
        """
        num_points = collision_points.shape[0]

        # 未启用自适应时,所有点使用固定步数
        if not self.config.enable_adaptive_bisection:
            return np.full(num_points, self.config.bisection_steps, dtype=np.int32)

        min_steps = self.config.min_bisection_steps
        max_steps = self.config.max_bisection_steps

        # 计算每个碰撞点到中心的距离
        distances = np.linalg.norm(collision_points - center, axis=1)

        # 归一化距离指标 ∈ (0, 1]
        max_dist = np.max(distances) if len(distances) > 0 else 1.0
        normalized_dist = distances / max(max_dist, 1e-8)

        # 线性映射: 距离越大 → 步数越多(远距离区间大,需要更高精度)
        # normalized_dist=0 (最近) → min_steps
        # normalized_dist=1 (最远) → max_steps
        steps = min_steps + (max_steps - min_steps) * normalized_dist

        # 钳位到有效范围并转为整数
        steps = np.clip(steps, min_steps, max_steps).astype(np.int32)

        return steps

    def _bisection_single(
        self,
        collision_point: np.ndarray,
        center: np.ndarray,
        checker: CollisionCheckerAdapter,
        steps: int = None
    ) -> np.ndarray:
        """
        对单个碰撞点执行二分搜索

        Args:
            collision_point: 碰撞点
            center: 椭球体中心(无碰撞)
            checker: 碰撞检测器
            steps: 二分搜索步数,若为None则使用配置默认值

        Returns:
            边界点
        """
        if steps is None:
            steps = self.config.bisection_steps

        # 初始化区间（不copy，二分搜索中a/b会被重新赋值）
        a = center
        b = collision_point

        # 执行二分搜索
        for _ in range(steps):
            # 计算中点
            m = (a + b) / 2.0

            # 检查中点是否无碰撞
            if checker.check_config_collision_free(m):
                # 中点无碰撞,更新a
                a = m
            else:
                # 中点有碰撞,更新b
                b = m

        # 返回边界点
        # a是最后一个无碰撞的点,b是第一个有碰撞的点
        # 为了安全,返回a(无碰撞点),并稍微往中心方向退一点
        # 这样可以确保边界点不会触碰障碍物

        # 安全边距: 根据分辨率动态调整
        # 默认至少留出1个栅格的安全距离
        safety_margin = max(0.01, 1 * 0.1)  # 至少0.01m或1个栅格
        
        direction = a - center
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
            boundary = a - safety_margin * direction
        else:
            boundary = a

        return boundary

    def search_boundary_with_ellipsoid(
        self,
        collision_points: np.ndarray,
        ellipsoid: 'Hyperellipsoid',
        checker: CollisionCheckerAdapter
    ) -> np.ndarray:
        """
        使用椭球体信息执行二分搜索

        Args:
            collision_points: 碰撞点数组
            ellipsoid: 椭球体对象
            checker: 碰撞检测器

        Returns:
            边界点数组
        """
        if not DRAKE_AVAILABLE:
            warnings.warn("Drake不可用,使用中心点")
            center = np.mean(collision_points, axis=0)
            return self.search_boundary(collision_points, center, checker)

        try:
            center = ellipsoid.center()
            return self.search_boundary(collision_points, center, checker)
        except Exception:
            # 如果获取中心失败,使用碰撞点的平均
            center = np.mean(collision_points, axis=0)
            return self.search_boundary(collision_points, center, checker)

    def __str__(self) -> str:
        """
        返回二分搜索器的字符串表示

        Returns:
            格式化的字符串
        """
        return (
            f"BisectionSearcher(\n"
            f"  bisection_steps={self.config.bisection_steps}\n"
            f"  adaptive={self.config.enable_adaptive_bisection}, "
            f"min={self.config.min_bisection_steps}, max={self.config.max_bisection_steps}\n"
            f"  parallel={self.config.enable_parallel_bisection}, "
            f"workers={self.config.parallel_bisection_workers}, "
            f"threshold={self.config.parallel_bisection_threshold}\n"
            f")"
        )
