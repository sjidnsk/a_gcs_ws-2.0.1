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
from .iriszo_sampler_jit import NUMBA_AVAILABLE, _hit_and_run_sample_jit, _sample_from_ellipsoid_jit


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
        self._jit_compiled = NUMBA_AVAILABLE  # JIT编译是否可用

    def sample(
        self,
        polyhedron: HPolyhedron,
        num_samples: int,
        starting_point: Optional[np.ndarray] = None,
        mix_steps: Optional[int] = None,
        A: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        ellipsoid_center: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        在多面体内均匀采样

        Args:
            polyhedron: 多面体(Drake HPolyhedron对象)
            num_samples: 采样点数量
            starting_point: 起始点(可选),必须在多面体内
            mix_steps: 混合步数(可选),None时使用默认值20
            A: 约束矩阵缓存(可选),传入时避免从Drake提取
            b: 约束向量缓存(可选),传入时避免从Drake提取
            ellipsoid_center: 椭球中心(可选),用作起始点候选

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

        effective_mix = mix_steps if mix_steps is not None else self.mix_steps

        # 使用缓存的A/b或从Drake提取
        if A is None or b is None:
            # 获取维度
            dim = polyhedron.ambient_dimension()
            # 一次性提取约束矩阵并缓存，避免后续重复调用 Drake C++ 接口
            A = polyhedron.A()
            b = polyhedron.b()
        else:
            dim = A.shape[1]

        # 找到起始点
        if starting_point is None:
            starting_point = self._find_starting_point(
                polyhedron, A, b, ellipsoid_center=ellipsoid_center
            )
        else:
            # 验证起始点在多面体内（优先使用numpy替代）
            if not self._point_in_set_numpy(starting_point, A, b):
                warnings.warn("起始点不在多面体内,将尝试找到新的起始点")
                starting_point = self._find_starting_point(
                    polyhedron, A, b, ellipsoid_center=ellipsoid_center
                )

        # 初始化采样点数组
        samples = np.zeros((num_samples, dim))
        current_point = starting_point.copy()

        # JIT加速路径：融合sample+hit_and_run_step+compute_interval为单一JIT函数
        if self.config.enable_numba_jit and self._jit_compiled:
            try:
                seed = self._next_seed()
                return _hit_and_run_sample_jit(
                    starting_point, A, b, num_samples, effective_mix, seed
                )
            except Exception:
                self._jit_compiled = False
                warnings.warn("JIT采样失败，回退到numpy向量化")

        # 生成采样点（numpy向量化回退路径）
        for i in range(num_samples):
            # 执行effective_mix次hit-and-run步骤
            for _ in range(effective_mix):
                current_point = self._hit_and_run_step(
                    current_point, A, b
                )

            # 保存采样点
            samples[i] = current_point.copy()

        return samples

    def _point_in_set_numpy(
        self,
        point: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        tolerance: Optional[float] = None
    ) -> bool:
        """
        使用numpy检查点是否在多面体内（替代Drake PointInSet）

        Args:
            point: 待检查的点
            A: 约束矩阵 (m × dim)
            b: 约束向量 (m,)
            tolerance: 数值容差，None时使用config中的值

        Returns:
            True如果点在多面体内
        """
        tol = tolerance if tolerance is not None else self.config.point_in_set_tolerance
        return bool(np.all(A @ point <= b + tol))

    def _find_starting_point(
        self,
        polyhedron: HPolyhedron,
        A: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        ellipsoid_center: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        找到多面体内的一个起始点

        优先使用椭球中心（已知在多面体内），回退到Chebyshev中心或边界框中心。

        Args:
            polyhedron: 多面体
            A: 多面体约束矩阵（可选，已缓存时传入避免重复调用）
            b: 多面体约束右端向量（可选，已缓存时传入避免重复调用）
            ellipsoid_center: 椭球中心（可选，已知在多面体内的点）

        Returns:
            起始点

        Raises:
            ValueError: 如果无法找到有效点
        """
        # 优先使用椭球中心（已知在多面体内）
        if ellipsoid_center is not None and A is not None and b is not None:
            if self._point_in_set_numpy(ellipsoid_center, A, b):
                return ellipsoid_center.copy()

        try:
            # 尝试使用Chebyshev中心
            center = polyhedron.ChebyshevCenter()
            if A is not None and b is not None:
                if self._point_in_set_numpy(center, A, b):
                    return center
            elif polyhedron.PointInSet(center):
                return center
        except Exception:
            pass

        # 如果Chebyshev中心不可用,尝试使用边界框中心
        try:
            # 获取约束矩阵（使用缓存或从多面体提取）
            if A is None or b is None:
                A = polyhedron.A()
                b = polyhedron.b()
            dim = A.shape[1]
            lb, ub = self._get_bounding_box(A, b, dim)
            center = (lb + ub) / 2.0
            if self._point_in_set_numpy(center, A, b):
                return center
        except Exception:
            pass

        raise ValueError("无法找到多面体内的有效起始点")

    def _get_bounding_box(
        self,
        A: np.ndarray,
        b: np.ndarray,
        dim: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取多面体的边界框（向量化实现）

        向量化原理：
            原实现双重循环: 对每个约束i和维度j, 根据A[i,j]正负更新ub[j]/lb[j]
            向量化: 用 numpy 广播一次性计算 b[:,None]/A, 再用条件操作分类取 min/max

        Args:
            A: 多面体约束矩阵 (m × dim)
            b: 多面体约束右端向量 (m,)
            dim: 空间维度

        Returns:
            (lower_bound, upper_bound) 元组
        """
        # 正系数: A[i,j] > 0 时 x_j <= b_i / A_ij
        pos_mask = A > 1e-10
        ratios_pos = np.where(pos_mask, b[:, None] / A, np.inf)
        ub = np.min(ratios_pos, axis=0)

        # 负系数: A[i,j] < 0 时 x_j >= b_i / A_ij
        neg_mask = A < -1e-10
        ratios_neg = np.where(neg_mask, b[:, None] / A, -np.inf)
        lb = np.max(ratios_neg, axis=0)

        return lb, ub

    def _hit_and_run_step(
        self,
        x: np.ndarray,
        A: np.ndarray,
        b: np.ndarray
    ) -> np.ndarray:
        """
        执行一次hit-and-run步骤

        Args:
            x: 当前点
            A: 多面体约束矩阵 (m × dim)
            b: 多面体约束右端向量 (m,)

        Returns:
            新采样点
        """
        dim = len(x)

        # 随机选择一个方向(从单位球均匀采样)
        d = self.random_state.randn(dim)
        d = d / np.linalg.norm(d)

        # 计算沿方向d的可行区间
        a, b_interval = self._compute_interval(x, d, A, b)

        # 在[a, b]上均匀采样
        t = self.random_state.uniform(a, b_interval)

        # 计算新点
        new_x = x + t * d

        return new_x

    def _compute_interval(
        self,
        x: np.ndarray,
        d: np.ndarray,
        A: np.ndarray,
        b: np.ndarray
    ) -> Tuple[float, float]:
        """
        计算沿方向d的可行区间（向量化实现）

        给定当前点x和方向d,计算参数t的范围[a, b],
        使得x + t*d在多面体内当且仅当t in [a, b]。

        向量化原理：
            原实现逐约束循环: c_i = A[i]·d, rhs_i = b[i] - A[i]·x
            向量化: c_all = A @ d, rhs_all = b - A @ x
            然后用 numpy 条件操作分类求上下界

        Args:
            x: 当前点
            d: 方向向量
            A: 多面体约束矩阵 (m × dim)
            b: 多面体约束右端向量 (m,)

        Returns:
            (a, b) 区间边界
        """
        # 一次性矩阵-向量乘法，替代逐约束 Python 循环
        # 约束: A_i * (x + t*d) <= b_i
        # 展开: A_i * x + t * (A_i * d) <= b_i
        # 令 c = A_i * d, rhs = b_i - A_i * x
        # 则 t * c <= rhs
        c_all = A @ d          # shape: (m,), 所有约束的 c 值
        rhs_all = b - A @ x    # shape: (m,), 所有约束的 rhs 值

        # 分类处理：c > 0 时 t <= rhs/c (上界), c < 0 时 t >= rhs/c (下界)
        pos_mask = c_all > 1e-10
        neg_mask = c_all < -1e-10

        # 上界: c > 0 时取 rhs/c, 否则置 inf (在 min 中被忽略)
        upper_bounds = np.where(pos_mask, rhs_all / c_all, np.inf)
        b_interval = float(np.min(upper_bounds))

        # 下界: c < 0 时取 rhs/c, 否则置 -inf (在 max 中被忽略)
        lower_bounds = np.where(neg_mask, rhs_all / c_all, -np.inf)
        a = float(np.max(lower_bounds))

        # 添加小的边界冗余,避免数值问题
        margin = 1e-6 * (b_interval - a) if np.isfinite(a) and np.isfinite(b_interval) else 1e-6
        a += margin
        b_interval -= margin

        return a, b_interval

    def _compute_interval_scalar(
        self,
        x: np.ndarray,
        d: np.ndarray,
        A: np.ndarray,
        b: np.ndarray
    ) -> Tuple[float, float]:
        """
        计算沿方向d的可行区间（原逐元素实现，用于调试/验证）

        Args:
            x: 当前点
            d: 方向向量
            A: 多面体约束矩阵 (m × dim)
            b: 多面体约束右端向量 (m,)

        Returns:
            (a, b) 区间边界
        """
        a = -np.inf
        b_interval = np.inf

        for i in range(A.shape[0]):
            c = np.dot(A[i], d)
            rhs = b[i] - np.dot(A[i], x)

            if abs(c) < 1e-10:
                continue
            elif c > 0:
                b_interval = min(b_interval, rhs / c)
            else:
                a = max(a, rhs / c)

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

        # JIT加速路径
        if self.config.enable_numba_jit and self._jit_compiled:
            try:
                seed = self._next_seed()
                return _sample_from_ellipsoid_jit(center, radius, num_samples, seed)
            except Exception:
                pass  # 静默回退到numpy路径

        for i in range(num_samples):
            # 从标准正态分布采样
            x = self.random_state.randn(dim)
            x = x / np.linalg.norm(x)

            # 从[0, 1]的均匀分布采样r^(1/dim)
            r = self.random_state.uniform(0, 1) ** (1.0 / dim)

            # 缩放到半径
            samples[i] = center + radius * r * x

        return samples

    def _next_seed(self) -> int:
        """
        生成下一个随机种子供JIT函数使用

        从RandomState生成种子，确保不同调用使用不同种子，
        同时受set_seed()控制。

        Returns:
            随机种子
        """
        return int(self.random_state.randint(0, 2**31))

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
