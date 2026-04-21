"""
IrisZo采样器Numba JIT加速模块

将Hit-and-Run采样器的核心循环融合为JIT编译函数，
消除Python解释器调度开销和小数组分配开销。

优化策略:
1. 融合sample()双重循环 + _hit_and_run_step() + _compute_interval()为单一JIT函数
2. 标量内联所有numpy小数组操作（randn、norm、矩阵乘法）
3. 逐约束标量循环计算可行区间，零临时数组分配
4. Numba不可用时透明退化为纯Python函数（无加速但无报错）

作者: Path Planning Team
"""

import numpy as np
from math import sqrt, isfinite

# Numba JIT编译加速（可选，复用项目已有模式）
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    jit = lambda *args, **kwargs: lambda f: f


@jit(nopython=True, cache=True)
def _hit_and_run_sample_jit(
    starting_point: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    num_samples: int,
    mix_steps: int,
    seed: int
) -> np.ndarray:
    """
    Hit-and-Run采样JIT融合函数

    融合sample()双重循环 + _hit_and_run_step() + _compute_interval()，
    所有操作内联为标量运算，零临时数组分配。

    Args:
        starting_point: 起始点 (dim,)，必须在多面体内
        A: 约束矩阵 (m, dim) float64
        b: 约束向量 (m,) float64
        num_samples: 采样点数量
        mix_steps: 混合步数
        seed: 随机种子

    Returns:
        采样点数组 (num_samples, dim)
    """
    dim = starting_point.shape[0]
    m = A.shape[0]
    samples = np.zeros((num_samples, dim))

    # 初始化随机数生成器
    np.random.seed(seed)

    # 复制起始点作为当前点
    current = np.zeros(dim)
    for j in range(dim):
        current[j] = starting_point[j]

    for i in range(num_samples):
        for _ in range(mix_steps):
            # --- 内联 _hit_and_run_step ---

            # 随机方向：从标准正态采样并归一化
            d = np.random.randn(dim)
            norm_d = 0.0
            for j in range(dim):
                norm_d += d[j] * d[j]
            norm_d = sqrt(norm_d)
            for j in range(dim):
                d[j] = d[j] / norm_d

            # --- 内联 _compute_interval（标量循环）---
            # 约束: A_i * (x + t*d) <= b_i
            # 展开: t * (A_i * d) <= b_i - A_i * x
            # 令 c = A_i * d, rhs = b_i - A_i * x
            # c > 0: t <= rhs/c (上界)
            # c < 0: t >= rhs/c (下界)

            a_val = -1e30
            b_val = 1e30

            for k in range(m):
                # 标量内积: c = A[k] · d
                c = 0.0
                rhs = b[k]
                for j in range(dim):
                    c += A[k, j] * d[j]
                    rhs -= A[k, j] * current[j]

                if c > 1e-10:
                    ratio = rhs / c
                    if ratio < b_val:
                        b_val = ratio
                elif c < -1e-10:
                    ratio = rhs / c
                    if ratio > a_val:
                        a_val = ratio

            # 边界冗余（与numpy版本一致）
            if isfinite(a_val) and isfinite(b_val):
                margin = 1e-6 * (b_val - a_val)
            else:
                margin = 1e-6
            a_val += margin
            b_val -= margin

            # 在[a, b]上均匀采样
            t = np.random.uniform(a_val, b_val)

            # 更新当前点: current = current + t * d
            for j in range(dim):
                current[j] = current[j] + t * d[j]

        # 保存采样点
        for j in range(dim):
            samples[i, j] = current[j]

    return samples


@jit(nopython=True, cache=True)
def _sample_from_ellipsoid_jit(
    center: np.ndarray,
    radius: float,
    num_samples: int,
    seed: int
) -> np.ndarray:
    """
    椭球体均匀采样JIT函数

    Args:
        center: 椭球体中心 (dim,)
        radius: 椭球体半径
        num_samples: 采样点数量
        seed: 随机种子

    Returns:
        采样点数组 (num_samples, dim)
    """
    dim = center.shape[0]
    samples = np.zeros((num_samples, dim))

    np.random.seed(seed)

    for i in range(num_samples):
        # 从标准正态分布采样方向
        x = np.random.randn(dim)
        norm_x = 0.0
        for j in range(dim):
            norm_x += x[j] * x[j]
        norm_x = sqrt(norm_x)
        for j in range(dim):
            x[j] = x[j] / norm_x

        # 从[0, 1]均匀分布采样 r^(1/dim)
        r = np.random.uniform(0.0, 1.0) ** (1.0 / dim)

        # 缩放到半径: center + radius * r * x
        for j in range(dim):
            samples[i, j] = center[j] + radius * r * x[j]

    return samples
