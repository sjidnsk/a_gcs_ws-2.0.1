"""
自定义IrisZo算法采样器模块

实现Hit-and-Run采样器,在多面体内进行均匀采样。

作者: Path Planning Team
"""

import numpy as np
from typing import Optional, Tuple
import warnings

try:
    from pydrake.geometry.optimization import HPolyhedron
    DRAKE_AVAILABLE = True
except ImportError:
    DRAKE_AVAILABLE = False
    HPolyhedron = None

from ..config.iriszo_config import IrisZoConfig


class HitAndRunSampler:
    """
    Hit-and-Run采样器

    实现在多面体内的均匀采样,使用hit-and-run算法。

    Hit-and-Run算法原理:
        1. 从当前点x出发
        2. 随机选择一个方向d(从单位球均匀采样)
        3. 计算沿方向d的可行区间[a, b]
        4. 在[a, b]上均匀采样得到新点x'
        5. 重复步骤2-4直到达到混合步数

    该算法生成的采样点服从多面体内的均匀分布。

    Attributes:
        config: 配置参数
        mix_steps: 混合步数,控制采样点的随机性
        random_state: 随机数生成器

    Example:
        >>> from pydrake.geometry.optimization import HPolyhedron
        >>> import numpy as np
        >>>
        >>> # 创建一个单位正方形
        >>> lb = np.array([0.0, 0.0])
        >>> ub = np.array([1.0, 1.0])
        >>> polyhedron = HPolyhedron.MakeBox(lb, ub)
        >>>
        >>> # 创建采样器
        >>> sampler = HitAndRunSampler(config)
        >>>
        >>> # 采样100个点
        >>> samples = sampler.sample(polyhedron, num_samples=100)
    """

    def __init__(self, config: IrisZoConfig, seed: Optional[int] = None):
        """
        初始化Hit-and-Run采样器

        Args:
            config: 配置参数
            seed: 随机种子,用于可重复性
        """
        self.config = config
        self.mix_steps = 20  # 混合步数,经验值
        self.random_state = np.random.RandomState(seed)

    def sample(
        self,
        polyhedron: HPolyhedron,
        num_samples: int,
        starting_point: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        在多面体内均匀采样

        Args:
            polyhedron: 多面体(Drake HPolyhedron对象)
            num_samples: 采样点数量
            starting_point: 起始点(可选),必须在多面体内

        Returns:
            采样点数组,shape=(num_samples, dim)

        Raises:
            RuntimeError: 如果Drake不可用
            ValueError: 如果无法找到有效的起始点

        Example:
            >>> samples = sampler.sample(polyhedron, num_samples=100)
            >>> print(f"采样点形状: {samples.shape}")
        """
        if not DRAKE_AVAILABLE:
            raise RuntimeError("Drake不可用,无法执行采样")

        # 获取维度
        dim = polyhedron.ambient_dimension()

        # 找到起始点
        if starting_point is None:
            starting_point = self._find_starting_point(polyhedron)
        else:
            # 验证起始点在多面体内
            if not polyhedron.PointInSet(starting_point):
                warnings.warn("起始点不在多面体内,将尝试找到新的起始点")
                starting_point = self._find_starting_point(polyhedron)

        # 初始化采样点数组
        samples = np.zeros((num_samples, dim))
        current_point = starting_point.copy()

        # 生成采样点
        for i in range(num_samples):
            # 执行mix_steps次hit-and-run步骤
            for _ in range(self.mix_steps):
                current_point = self._hit_and_run_step(
                    current_point, polyhedron
                )

            # 保存采样点
            samples[i] = current_point.copy()

        return samples

    def _find_starting_point(self, polyhedron: HPolyhedron) -> np.ndarray:
        """
        找到多面体内的一个起始点

        使用Chebyshev中心作为起始点。

        Args:
            polyhedron: 多面体

        Returns:
            起始点

        Raises:
            ValueError: 如果无法找到有效点
        """
        try:
            # 尝试使用Chebyshev中心
            center = polyhedron.ChebyshevCenter()
            if polyhedron.PointInSet(center):
                return center
        except Exception:
            pass

        # 如果Chebyshev中心不可用,尝试使用边界框中心
        try:
            # 获取边界框
            lb, ub = self._get_bounding_box(polyhedron)
            center = (lb + ub) / 2.0
            if polyhedron.PointInSet(center):
                return center
        except Exception:
            pass

        raise ValueError("无法找到多面体内的有效起始点")

    def _get_bounding_box(
        self,
        polyhedron: HPolyhedron
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取多面体的边界框

        Args:
            polyhedron: 多面体

        Returns:
            (lower_bound, upper_bound) 元组
        """
        # 获取多面体的A和b矩阵(Ax <= b)
        A = polyhedron.A()
        b = polyhedron.b()
        dim = polyhedron.ambient_dimension()

        # 计算每个维度的边界
        lb = np.full(dim, -np.inf)
        ub = np.full(dim, np.inf)

        for i in range(A.shape[0]):
            for j in range(dim):
                if A[i, j] > 1e-10:
                    # x_j <= b_i / A_ij
                    ub[j] = min(ub[j], b[i] / A[i, j])
                elif A[i, j] < -1e-10:
                    # x_j >= b_i / A_ij
                    lb[j] = max(lb[j], b[i] / A[i, j])

        return lb, ub

    def _hit_and_run_step(
        self,
        x: np.ndarray,
        polyhedron: HPolyhedron
    ) -> np.ndarray:
        """
        执行一次hit-and-run步骤

        Args:
            x: 当前点
            polyhedron: 多面体

        Returns:
            新采样点
        """
        dim = len(x)

        # 随机选择一个方向(从单位球均匀采样)
        d = self.random_state.randn(dim)
        d = d / np.linalg.norm(d)

        # 计算沿方向d的可行区间
        a, b = self._compute_interval(x, d, polyhedron)

        # 在[a, b]上均匀采样
        t = self.random_state.uniform(a, b)

        # 计算新点
        new_x = x + t * d

        return new_x

    def _compute_interval(
        self,
        x: np.ndarray,
        d: np.ndarray,
        polyhedron: HPolyhedron
    ) -> Tuple[float, float]:
        """
        计算沿方向d的可行区间

        给定当前点x和方向d,计算参数t的范围[a, b],
        使得x + t*d在多面体内当且仅当t in [a, b]。

        Args:
            x: 当前点
            d: 方向向量
            polyhedron: 多面体

        Returns:
            (a, b) 区间边界
        """
        # 获取多面体的A和b矩阵(Ax <= b)
        A = polyhedron.A()
        b = polyhedron.b()

        # 初始化区间
        a = -np.inf
        b_interval = np.inf

        # 对每个约束计算t的范围
        for i in range(A.shape[0]):
            # 约束: A_i * (x + t*d) <= b_i
            # 展开: A_i * x + t * (A_i * d) <= b_i
            # 令 c = A_i * d, rhs = b_i - A_i * x
            # 则 t * c <= rhs

            c = np.dot(A[i], d)
            rhs = b[i] - np.dot(A[i], x)

            if abs(c) < 1e-10:
                # c ≈ 0,约束变为 0 <= rhs
                # 如果rhs < 0,则无解(不应该发生,因为x在多面体内)
                continue
            elif c > 0:
                # t <= rhs / c
                b_interval = min(b_interval, rhs / c)
            else:  # c < 0
                # t >= rhs / c
                a = max(a, rhs / c)

        # 添加小的边界冗余,避免数值问题
        margin = 1e-6 * (b_interval - a) if np.isfinite(a) and np.isfinite(b_interval) else 1e-6
        a += margin
        b_interval -= margin

        return a, b_interval

    def sample_from_ellipsoid(
        self,
        center: np.ndarray,
        radius: float,
        num_samples: int
    ) -> np.ndarray:
        """
        从椭球体内均匀采样

        生成一个球形区域内的均匀采样点。

        Args:
            center: 椭球体中心
            radius: 椭球体半径
            num_samples: 采样点数量

        Returns:
            采样点数组,shape=(num_samples, dim)

        Example:
            >>> center = np.array([0.5, 0.5])
            >>> samples = sampler.sample_from_ellipsoid(center, 0.1, 50)
        """
        dim = len(center)
        samples = np.zeros((num_samples, dim))

        for i in range(num_samples):
            # 从标准正态分布采样
            x = self.random_state.randn(dim)
            x = x / np.linalg.norm(x)

            # 从[0, 1]的均匀分布采样r^(1/dim)
            r = self.random_state.uniform(0, 1) ** (1.0 / dim)

            # 缩放到半径
            samples[i] = center + radius * r * x

        return samples

    def set_seed(self, seed: int) -> None:
        """
        设置随机种子

        Args:
            seed: 随机种子
        """
        self.random_state = np.random.RandomState(seed)

    def __str__(self) -> str:
        """
        返回采样器的字符串表示

        Returns:
            格式化的字符串
        """
        return (
            f"HitAndRunSampler(\n"
            f"  mix_steps={self.mix_steps}\n"
            f")"
        )
