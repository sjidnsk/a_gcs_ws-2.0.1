"""
IrisNp 工具函数模块

包含碰撞检测、几何计算等辅助函数。

作者: Path Planning Team
"""

import numpy as np
from typing import List, Tuple, Optional

from config.iris import IrisNpConfig
from .iris_np_region_data import IrisNpRegion
from .iris_np_collision import SimpleCollisionCheckerForIrisNp


def check_region_collision_optimized(
    config: IrisNpConfig,
    checker: SimpleCollisionCheckerForIrisNp,
    region: IrisNpRegion,
    resolution: float
) -> bool:
    """
    优化的区域碰撞检测

    使用批量检测和早期退出策略，并应用安全裕度
    """
    # 获取安全裕度
    safety_margin = config.configuration_space_margin

    # 策略1: 快速边界检查（带安全裕度）
    if check_boundary_collision_fast(checker, region, safety_margin):
        return True

    # 策略2: 内部采样检查（使用批量检测）
    if config.use_batch_collision_check:
        if check_interior_collision_batch(config, checker, region, resolution, safety_margin):
            return True
    else:
        if check_interior_collision_sequential(config, checker, region, resolution, safety_margin):
            return True

    return False


def check_boundary_collision_fast(
    checker: SimpleCollisionCheckerForIrisNp,
    region: IrisNpRegion,
    safety_margin: float = 0.0
) -> bool:
    """
    快速边界碰撞检测（带安全裕度）- 增强版

    修复漏洞1: 不仅检查顶点,还检查边上的采样点
    防止细长障碍物穿过边但未碰到顶点的情况
    """
    # 获取排序后的顶点
    vertices = region.get_vertices_ordered()
    if len(vertices) < 3:
        return False

    # 1. 检查所有顶点
    for vertex in vertices:
        if checker.check_collision(vertex, safety_margin):
            return True

    # 2. 检查每条边上的采样点
    # 根据边的长度自适应确定采样点数量
    for i in range(len(vertices)):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % len(vertices)]

        # 计算边的长度
        edge_length = np.linalg.norm(v2 - v1)

        # 根据边长和分辨率确定采样点数量
        # 每个分辨率距离至少采样一个点,最少3个点,最多20个点
        num_edge_samples = max(3, min(20, int(edge_length / checker.resolution) + 1))

        # 在边上均匀采样
        for t in np.linspace(0, 1, num_edge_samples, endpoint=False):
            # 跳过起点(已经检查过顶点)
            if t == 0:
                continue

            # 计算采样点
            point = v1 + t * (v2 - v1)

            # 检查碰撞
            if checker.check_collision(point, safety_margin):
                return True

    return False


def check_interior_collision_batch(
    config: IrisNpConfig,
    checker: SimpleCollisionCheckerForIrisNp,
    region: IrisNpRegion,
    resolution: float,
    safety_margin: float = 0.0
) -> bool:
    """
    批量内部碰撞检测（优化版本）- 增强版

    修复漏洞2: 根据区域大小自适应调整采样密度
    确保小障碍物也能被检测到
    """
    # 计算区域的边界框
    x_min = np.min(region.vertices[:, 0])
    x_max = np.max(region.vertices[:, 0])
    y_min = np.min(region.vertices[:, 1])
    y_max = np.max(region.vertices[:, 1])

    # 计算区域面积
    area = region.area

    # 根据区域面积和分辨率自适应确定采样密度
    # 目标: 每个分辨率单位的面积至少有一个采样点
    # 公式: grid_size = sqrt(area / (resolution^2 * target_density))
    # target_density: 每个网格单元期望的采样点数,设为0.5(每2个网格1个点)
    target_density = 0.5
    min_samples = 10  # 最少采样点数
    max_samples = 100  # 最多采样点数

    # 计算理想的采样点数
    ideal_samples = area / (resolution * resolution * target_density)
    num_samples = int(max(min_samples, min(max_samples, ideal_samples)))

    # 计算网格大小
    grid_size = int(np.sqrt(num_samples))
    grid_size = max(5, grid_size)  # 至少5x5

    x_samples = np.linspace(x_min, x_max, grid_size)
    y_samples = np.linspace(y_min, y_max, grid_size)

    # 创建网格点
    xx, yy = np.meshgrid(x_samples, y_samples)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    # 过滤出区域内的点（向量化）
    # 使用contains方法批量检查
    inside_mask = np.array([region.contains(point) for point in points])
    inside_points = points[inside_mask]

    if len(inside_points) == 0:
        return False

    # 批量碰撞检测
    collision_results = checker.check_collision_batch(inside_points, safety_margin)

    # 如果有任何碰撞,返回True
    return np.any(collision_results)


def check_interior_collision_sequential(
    config: IrisNpConfig,
    checker: SimpleCollisionCheckerForIrisNp,
    region: IrisNpRegion,
    resolution: float,
    safety_margin: float = 0.0
) -> bool:
    """
    顺序内部碰撞检测（带安全裕度）- 增强版

    修复漏洞2: 根据区域大小自适应调整采样密度
    确保小障碍物也能被检测到
    """
    # 计算区域的边界框
    x_min = np.min(region.vertices[:, 0])
    x_max = np.max(region.vertices[:, 0])
    y_min = np.min(region.vertices[:, 1])
    y_max = np.max(region.vertices[:, 1])

    # 计算区域面积
    area = region.area

    # 根据区域面积和分辨率自适应确定采样密度
    target_density = 0.5
    min_samples = 10
    max_samples = 100

    ideal_samples = area / (resolution * resolution * target_density)
    num_samples = int(max(min_samples, min(max_samples, ideal_samples)))

    grid_size = int(np.sqrt(num_samples))
    grid_size = max(5, grid_size)

    x_samples = np.linspace(x_min, x_max, grid_size)
    y_samples = np.linspace(y_min, y_max, grid_size)

    for x in x_samples:
        for y in y_samples:
            point = np.array([x, y])

            if region.contains(point):
                if checker.check_collision(point, safety_margin):
                    return True

    return False


def compute_polyhedron_vertices_optimized(
    A: np.ndarray,
    b: np.ndarray
) -> np.ndarray:
    """
    优化的多面体顶点计算

    使用向量化操作和早期退出策略
    """
    n_constraints = A.shape[0]

    if n_constraints < 3:
        return np.array([])

    # 预分配空间
    vertices = []

    # 向量化计算所有交点
    for i in range(n_constraints):
        for j in range(i + 1, n_constraints):
            # 求解两条直线的交点
            A_sub = np.vstack([A[i], A[j]])

            # 快速检查行列式是否为0
            det = A_sub[0, 0] * A_sub[1, 1] - A_sub[0, 1] * A_sub[1, 0]
            if abs(det) < 1e-10:
                continue

            try:
                # 使用克莱姆法则（更快）
                x = np.array([
                    (b[i] * A_sub[1, 1] - b[j] * A_sub[0, 1]) / det,
                    (A_sub[0, 0] * b[j] - A_sub[1, 0] * b[i]) / det
                ])

                # 检查点是否满足所有约束（向量化）
                if np.all(A @ x <= b + 1e-6):
                    vertices.append(x)
            except (np.linalg.LinAlgError, FloatingPointError):
                pass

    if len(vertices) == 0:
        return np.array([])

    vertices = np.array(vertices)

    # 去除重复点（使用更高效的方法）
    vertices = np.unique(np.round(vertices, decimals=6), axis=0)

    # 按角度排序顶点
    if len(vertices) >= 3:
        centroid = np.mean(vertices, axis=0)
        angles = np.arctan2(
            vertices[:, 1] - centroid[1],
            vertices[:, 0] - centroid[0]
        )
        sorted_indices = np.argsort(angles)
        vertices = vertices[sorted_indices]

    return vertices


def compute_polygon_area(vertices: np.ndarray) -> float:
    """计算多边形面积（鞋带公式）"""
    n = len(vertices)
    if n < 3:
        return 0.0

    # 按角度排序顶点
    centroid = np.mean(vertices, axis=0)
    angles = np.arctan2(
        vertices[:, 1] - centroid[1],
        vertices[:, 0] - centroid[0]
    )
    sorted_indices = np.argsort(angles)
    sorted_vertices = vertices[sorted_indices]

    # 计算面积
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += sorted_vertices[i, 0] * sorted_vertices[j, 1]
        area -= sorted_vertices[j, 0] * sorted_vertices[i, 1]

    return abs(area) / 2.0


def quick_boundary_collision_check(
    config: IrisNpConfig,
    checker: SimpleCollisionCheckerForIrisNp,
    center: np.ndarray,
    direction: np.ndarray,
    distance: float,
    resolution: float
) -> bool:
    """
    快速边界碰撞检查

    只检查边界点是否碰撞，而不是整个区域
    这避免了创建完整的区域对象，大幅减少内存开销

    Args:
        config: 配置参数
        checker: 碰撞检测器
        center: 中心点
        direction: 方向向量
        distance: 距离
        resolution: 分辨率

    Returns:
        True如果边界点碰撞，False否则
    """
    # 计算边界点
    boundary_point = center + direction * distance

    # 检查边界点是否碰撞
    safety_margin = config.configuration_space_margin
    return checker.check_collision(boundary_point, safety_margin)


def estimate_area_fast(
    distances: np.ndarray,
    directions: np.ndarray
) -> float:
    """
    快速估算区域面积（不创建区域对象）

    使用多边形面积公式估算，避免创建完整的区域对象

    Args:
        distances: 每个方向的距离
        directions: 方向向量数组

    Returns:
        估算的面积
    """
    # 计算边界点
    num_directions = len(directions)
    boundary_points = np.zeros((num_directions, 2))
    for i in range(num_directions):
        boundary_points[i] = directions[i] * distances[i]

    # 按角度排序
    angles = np.arctan2(boundary_points[:, 1], boundary_points[:, 0])
    sorted_indices = np.argsort(angles)
    sorted_points = boundary_points[sorted_indices]

    # 使用鞋带公式计算面积
    n = len(sorted_points)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += sorted_points[i, 0] * sorted_points[j, 1]
        area -= sorted_points[j, 0] * sorted_points[i, 1]

    return abs(area) / 2.0
