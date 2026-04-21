"""
自定义IrisZo算法分离超平面生成模块

实现分离超平面生成器,为每个碰撞点生成对应的分离超平面。

作者: Path Planning Team
"""

import numpy as np
from typing import List, Tuple, Optional
import warnings

try:
    from pydrake.geometry.optimization import HPolyhedron, Hyperellipsoid
    DRAKE_AVAILABLE = True
except ImportError:
    DRAKE_AVAILABLE = False
    HPolyhedron = None
    Hyperellipsoid = None

from ..config.iriszo_config import IrisZoConfig


class SeparatingHyperplaneGenerator:
    """
    分离超平面生成器

    为每个碰撞点生成对应的分离超平面,将碰撞点与当前多面体分离。

    分离超平面生成算法原理:
        1. 对于碰撞点p和椭球体E(中心c)
        2. 计算从椭球体中心c到p的方向d = p - c
        3. 归一化方向d得到法向量n = d / ||d||
        4. 计算超平面偏移b = n · p
        5. 超平面方程: n · x ≤ b

    该超平面将碰撞点p与椭球体E分离,确保p在超平面外侧。

    Attributes:
        config: 配置参数

    Example:
        >>> generator = SeparatingHyperplaneGenerator(config)
        >>> hyperplanes = generator.generate(
        ...     collision_points, ellipsoid, current_polyhedron
        ... )
    """

    def __init__(self, config: IrisZoConfig):
        """
        初始化分离超平面生成器

        Args:
            config: 配置参数
        """
        self.config = config

    def generate(
        self,
        collision_points: np.ndarray,
        ellipsoid: 'Hyperellipsoid',
        current_polyhedron: HPolyhedron
    ) -> List[Tuple[np.ndarray, float]]:
        """
        生成分离超平面

        Args:
            collision_points: 碰撞点数组,shape=(N, dim)
            ellipsoid: 当前内接椭球体
            current_polyhedron: 当前多面体

        Returns:
            超平面列表,每个元素为(法向量n, 偏移b)

        Example:
            >>> hyperplanes = generator.generate(
            ...     collision_points, ellipsoid, polyhedron
            ... )
        """
        if not DRAKE_AVAILABLE:
            warnings.warn("Drake不可用,无法生成超平面")
            return []

        # 获取椭球体中心
        try:
            ellipsoid_center = ellipsoid.center()
        except Exception:
            warnings.warn("无法获取椭球体中心,使用碰撞点平均")
            ellipsoid_center = np.mean(collision_points, axis=0)

        # 向量化生成超平面
        d = collision_points - ellipsoid_center  # (N, dim)
        norms = np.linalg.norm(d, axis=1, keepdims=True)  # (N, 1)

        # 处理零向量：碰撞点与中心重合时使用随机方向
        zero_mask = (norms < 1e-10).squeeze()
        if np.any(zero_mask):
            num_zero = int(np.sum(zero_mask))
            d[zero_mask] = np.random.randn(num_zero, d.shape[1])
            norms[zero_mask] = np.linalg.norm(d[zero_mask], axis=1, keepdims=True)

        n = d / norms  # (N, dim) 法向量
        b = np.sum(n * collision_points, axis=1)  # (N,) 偏移

        hyperplanes = [(n[i], b[i]) for i in range(len(n))]

        # 移除冗余超平面
        non_redundant = self._remove_redundant(hyperplanes, current_polyhedron)

        if self.config.verbose and len(hyperplanes) != len(non_redundant):
            print(f"生成{len(hyperplanes)}个超平面,移除冗余后剩余{len(non_redundant)}个")

        return non_redundant

    def _generate_single(
        self,
        collision_point: np.ndarray,
        ellipsoid_center: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        为单个碰撞点生成超平面

        Args:
            collision_point: 碰撞点
            ellipsoid_center: 椭球体中心

        Returns:
            (法向量n, 偏移b)
        """
        # 计算方向向量
        d = collision_point - ellipsoid_center

        # 归一化得到法向量
        norm = np.linalg.norm(d)
        if norm < 1e-10:
            # 如果碰撞点与中心重合,使用随机方向
            warnings.warn("碰撞点与椭球体中心重合,使用随机方向")
            d = np.random.randn(len(d))
            norm = np.linalg.norm(d)

        n = d / norm

        # 计算偏移
        b = np.dot(n, collision_point)

        return n, b

    def _remove_redundant(
        self,
        hyperplanes: List[Tuple[np.ndarray, float]],
        current_polyhedron: HPolyhedron
    ) -> List[Tuple[np.ndarray, float]]:
        """
        移除冗余超平面

        检查每个超平面是否被其他超平面隐含,移除不影响多面体形状的超平面。

        Args:
            hyperplanes: 超平面列表
            current_polyhedron: 当前多面体

        Returns:
            非冗余超平面列表
        """
        if len(hyperplanes) == 0:
            return []

        # 简单策略:保留所有超平面
        # TODO: 实现更智能的冗余检测
        # 可以通过求解线性规划来检查超平面是否冗余

        return hyperplanes

    def generate_from_points_and_center(
        self,
        collision_points: np.ndarray,
        center: np.ndarray
    ) -> List[Tuple[np.ndarray, float]]:
        """
        从碰撞点和中心点生成超平面

        Args:
            collision_points: 碰撞点数组
            center: 中心点

        Returns:
            超平面列表
        """
        hyperplanes = []
        for collision_point in collision_points:
            n, b = self._generate_single(collision_point, center)
            hyperplanes.append((n, b))
        return hyperplanes

    def update_polyhedron_cache(
        self,
        A_cached: np.ndarray,
        b_cached: np.ndarray,
        hyperplanes: List[Tuple[np.ndarray, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用新超平面增量更新A/b缓存（纯numpy，无Drake调用）

        Args:
            A_cached: 当前缓存的约束矩阵 (m × dim)
            b_cached: 当前缓存的约束向量 (m,)
            hyperplanes: 新超平面列表

        Returns:
            (A_updated, b_updated) 更新后的缓存
        """
        if len(hyperplanes) == 0:
            return A_cached, b_cached

        # 向量化构建新约束矩阵
        A_new = np.array([n for n, b in hyperplanes])
        b_new = np.array([b for n, b in hyperplanes])

        # 增量合并（纯numpy，无Drake调用）
        A_updated = np.vstack([A_cached, A_new])
        b_updated = np.concatenate([b_cached, b_new])

        return A_updated, b_updated

    def update_polyhedron(
        self,
        current_polyhedron: HPolyhedron,
        hyperplanes: List[Tuple[np.ndarray, float]]
    ) -> Optional[HPolyhedron]:
        """
        使用新超平面更新多面体

        Args:
            current_polyhedron: 当前多面体
            hyperplanes: 新超平面列表

        Returns:
            更新后的多面体,如果失败则返回None

        Example:
            >>> new_polyhedron = generator.update_polyhedron(
            ...     current_polyhedron, hyperplanes
            ... )
        """
        if not DRAKE_AVAILABLE:
            warnings.warn("Drake不可用,无法更新多面体")
            return None

        if len(hyperplanes) == 0:
            return current_polyhedron

        try:
            # 获取当前多面体的约束
            A_current = current_polyhedron.A()
            b_current = current_polyhedron.b()

            # 向量化构建新约束矩阵
            A_new = np.array([n for n, b in hyperplanes])
            b_new = np.array([b for n, b in hyperplanes])

            # 合并约束
            A_combined = np.vstack([A_current, A_new])
            b_combined = np.concatenate([b_current, b_new])

            # 创建新多面体
            new_polyhedron = HPolyhedron(A_combined, b_combined)

            return new_polyhedron

        except Exception as e:
            warnings.warn(f"更新多面体失败: {e}")
            return current_polyhedron

    def __str__(self) -> str:
        """
        返回超平面生成器的字符串表示

        Returns:
            格式化的字符串
        """
        return "SeparatingHyperplaneGenerator()"
