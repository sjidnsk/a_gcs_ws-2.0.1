"""
工具函数模块

提供阿克曼GCS轨迹优化器所需的工具函数。
"""

import numpy as np


__all__ = ['sample_unit_sphere']


def sample_unit_sphere(dim, num_samples):
    """
    在 n 维单位超球面上生成均匀采样点。

    Args:
        dim (int): 空间维度 n。
        num_samples (int): 期望的采样点数量。

    Returns:
        np.ndarray: 形状为 (dim, num_samples) 的数组，每一列是一个单位超球面上的点。
    """
    # 在 n 维空间中，从标准正态分布 N(0, I) 采样
    # 然后将每个样本向量归一化到单位长度
    # 这样得到的点在单位超球面上是均匀分布的

    # 生成 (dim, num_samples) 的高斯随机矩阵
    gaussian_samples = np.random.normal(size=(dim, num_samples))

    # 计算每个样本向量的 L2 范数 (沿 dim 轴)
    norms = np.linalg.norm(gaussian_samples, axis=0, keepdims=True) # shape: (1, num_samples)

    # 避免除零错误（虽然概率极低）
    norms = np.where(norms == 0, 1.0, norms)

    # 将每个向量除以其范数，得到单位向量
    unit_vectors = gaussian_samples / norms # Broadcasting: (dim, num_samples) / (1, num_samples)

    return unit_vectors
