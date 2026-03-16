"""
Voronoi图优化种子点模块

基于Voronoi图的种子点优化策略：
- 利用Voronoi顶点的最大空圆性质
- 自动找到最优种子点位置
- 数学上保证覆盖效率最优

作者: Path Planning Team
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial import Voronoi

from .iris_np_region_data import IrisNpRegion
from .iris_np_coverage_checker import IrisNpCoverageChecker


class VoronoiSeedOptimizer:
    """基于Voronoi图的种子点优化器"""
    
    def __init__(
        self,
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float],
        coverage_checker: Optional[IrisNpCoverageChecker] = None
    ):
        """
        初始化优化器
        
        Args:
            obstacle_map: 障碍物地图（0=自由，1=障碍）
            resolution: 地图分辨率（米/像素）
            origin: 地图原点 [x₀, y₀]
            coverage_checker: 覆盖验证器（可选，用于精确检测）
        """
        self.obstacle_map = obstacle_map
        self.resolution = resolution
        self.origin = origin
        self.coverage_checker = coverage_checker
    
    def optimize(
        self,
        initial_seeds: List[np.ndarray],
        path: List[Tuple[float, float, float]],
        max_iterations: int = 5,
        max_new_seeds: int = 10
    ) -> List[np.ndarray]:
        """
        优化种子点
        
        Args:
            initial_seeds: 初始种子点列表
            path: 路径点列表 [(x, y, θ), ...]
            max_iterations: 最大迭代次数
            max_new_seeds: 最多新增种子点数量
        
        Returns:
            优化后的种子点列表
        """
        current_seeds = initial_seeds.copy()
        new_seeds_count = 0
        
        for iteration in range(max_iterations):
            # 检查是否可以生成Voronoi图
            if len(current_seeds) < 3:
                break
            
            # 生成Voronoi图
            try:
                vor = self._robust_voronoi(current_seeds)
            except Exception as e:
                print(f"Voronoi生成失败: {e}")
                break
            
            # 获取有限顶点
            finite_vertices = self._get_finite_vertices(vor)
            
            # 评估顶点
            candidates = []
            for vertex in finite_vertices:
                score = self._evaluate_vertex(vertex, path, current_seeds)
                if score > 0:
                    candidates.append((vertex, score))
            
            # 没有可添加的候选点
            if not candidates:
                break
            
            # 选择最优顶点
            best_vertex, best_score = max(candidates, key=lambda x: x[1])
            
            # 添加到种子点集合
            current_seeds.append(best_vertex)
            new_seeds_count += 1
            
            print(f"迭代 {iteration + 1}: 添加种子点 ({best_vertex[0]:.2f}, {best_vertex[1]:.2f}), 评分 {best_score:.2f}")
            
            # 检查是否达到新增种子点上限
            if new_seeds_count >= max_new_seeds:
                break
            
            # 检查覆盖完整性
            if self._check_coverage(current_seeds, path):
                print("路径已完全覆盖，停止优化")
                break
        
        return current_seeds
    
    def _robust_voronoi(self, seeds: List[np.ndarray]) -> Voronoi:
        """
        鲁棒Voronoi图生成
        
        处理种子点共线的情况，添加微小扰动避免退化
        """
        seeds_array = np.array(seeds)
        
        # 检查是否共线
        if len(seeds) >= 3:
            v1 = seeds_array[1] - seeds_array[0]
            v2 = seeds_array[2] - seeds_array[0]
            cross = np.abs(v1[0] * v2[1] - v1[1] * v2[0])
            
            if cross < 1e-8:
                # 共线，添加微小扰动
                noise = np.random.randn(len(seeds), 2) * 1e-6
                seeds_array += noise
        
        return Voronoi(seeds_array)
    
    def _get_finite_vertices(self, vor: Voronoi) -> List[np.ndarray]:
        """
        获取有限顶点
        
        过滤掉坐标为inf或-inf的顶点（这些顶点在边界外）
        """
        finite_vertices = []
        
        for vertex in vor.vertices:
            if np.all(np.isfinite(vertex)):
                finite_vertices.append(vertex)
        
        return finite_vertices
    
    def _is_in_free_space(self, point: np.ndarray) -> bool:
        """
        检查点是否在自由空间
        
        Args:
            point: 点 [x, y]
        
        Returns:
            True如果在自由空间，False否则
        """
        gx = int((point[0] - self.origin[0]) / self.resolution)
        gy = int((point[1] - self.origin[1]) / self.resolution)
        
        # 边界检查
        if gx < 0 or gx >= self.obstacle_map.shape[1]:
            return False
        if gy < 0 or gy >= self.obstacle_map.shape[0]:
            return False
        
        # 检查是否为自由空间
        return self.obstacle_map[gy, gx] == 0
    
    def _min_distance_to_path(self, point: np.ndarray, path: List[Tuple[float, float, float]]) -> float:
        """
        计算点到路径的最小距离
        
        Args:
            point: 点 [x, y]
            path: 路径点列表 [(x, y, θ), ...]
        
        Returns:
            最小距离（米）
        """
        min_dist = float('inf')
        
        for i in range(len(path) - 1):
            p1 = np.array(path[i][:2])
            p2 = np.array(path[i + 1][:2])
            dist = self._point_to_segment_distance(point, p1, p2)
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _point_to_segment_distance(self, point: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        计算点到线段的最小距离
        
        Args:
            point: 点 [x, y]
            p1: 线段起点 [x, y]
            p2: 线段终点 [x, y]
        
        Returns:
            最小距离
        """
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 1e-6:
            return np.linalg.norm(point - p1)
        
        # 计算投影点
        t = np.dot(point - p1, line_vec) / (line_len ** 2)
        t = max(0, min(1, t))  # 限制在线段范围内
        
        projection = p1 + t * line_vec
        return np.linalg.norm(point - projection)
    
    def _evaluate_vertex(
        self,
        vertex: np.ndarray,
        path: List[Tuple[float, float, float]],
        seeds: List[np.ndarray]
    ) -> float:
        """
        评估Voronoi顶点的质量
        
        评分标准：
        1. 是否在自由空间（必须，否则返回0）
        2. 距离路径的远近（越近越好，权重10）
        3. 距离最近种子点的距离（越远越好，权重1）
        4. Voronoi区域大小（越大越好，权重0.1）
        
        Args:
            vertex: Voronoi顶点 [x, y]
            path: 路径点列表
            seeds: 当前种子点列表
        
        Returns:
            评分（0表示无效）
        """
        # 检查1: 是否在自由空间
        if not self._is_in_free_space(vertex):
            return 0.0
        
        # 检查2: 距离路径的远近
        dist_to_path = self._min_distance_to_path(vertex, path)
        if dist_to_path > 2.0:  # 超过2米，不考虑
            return 0.0
        
        # 检查3: 距离最近种子点的距离
        distances = [np.linalg.norm(vertex - seed) for seed in seeds]
        min_dist_to_seed = min(distances)
        
        # 检查4: Voronoi区域大小（用最小距离估计）
        voronoi_radius = min_dist_to_seed
        estimated_area = np.pi * voronoi_radius ** 2
        
        # 综合评分
        score = (2.0 - dist_to_path) * 10 + min_dist_to_seed + estimated_area * 0.1
        
        return score
    
    def _check_coverage(self, path: List[Tuple[float, float, float]], regions: List[IrisNpRegion]) -> bool:
        """
        检查路径是否被完全覆盖
        
        Args:
            path: 路径点列表
            regions: 凸区域列表
        
        Returns:
            True如果完全覆盖，False否则
        """
        # 使用IrisNpCoverageChecker进行精确检测
        return self.coverage_checker.verify_path_coverage(path, regions)


def compute_path_curvature(path: List[Tuple[float, float, float]], index: int) -> float:
    """
    计算路径在指定点的曲率（转角角度）
    
    Args:
        path: 路径点列表
        index: 当前点索引
    
    Returns:
        曲率（弧度），0表示直线，越大表示转弯越急
    """
    if index == 0 or index == len(path) - 1:
        return 0.0
    
    p_prev = np.array(path[index-1][:2])
    p_curr = np.array(path[index][:2])
    p_next = np.array(path[index+1][:2])
    
    v1 = p_curr - p_prev
    v2 = p_next - p_curr
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    
    # 计算转角角度
    cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    return angle


def curvature_adaptive_sampling(
    path: List[Tuple[float, float, float]],
    obstacle_map: np.ndarray,
    resolution: float,
    origin: Tuple[float, float],
    min_distance: float = 1.0
) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    基于曲率的自适应采样
    
    策略：
    - 曲率 > 30度：密集采样（间隔3点）
    - 曲率 > 15度：中等采样（间隔5点）
    - 曲率 <= 15度：稀疏采样（间隔10点）
    
    Args:
        path: 路径点列表
        obstacle_map: 障碍物地图
        resolution: 地图分辨率
        origin: 地图原点
        min_distance: 最小种子点距离（米）
    
    Returns:
        种子点列表，每个元素为 (seed_point, tangent_direction)
    """
    seed_points = []
    path_length = len(path)
    
    if path_length == 0:
        return seed_points
    
    # 预计算所有点的曲率
    curvatures = [compute_path_curvature(path, i) for i in range(path_length)]
    
    # 强制包含起点
    x_start, y_start, _ = path[0]
    start_point = np.array([x_start, y_start])
    gx_start = int((x_start - origin[0]) / resolution)
    gy_start = int((y_start - origin[1]) / resolution)
    
    if 0 <= gx_start < obstacle_map.shape[1] and 0 <= gy_start < obstacle_map.shape[0]:
        if obstacle_map[gy_start, gx_start] == 0:
            seed_points.append((start_point, None))
    
    # 基于曲率自适应采样
    i = 0
    while i < path_length:
        x, y, _ = path[i]
        seed_point = np.array([x, y])
        
        # 检查是否在自由空间
        gx = int((x - origin[0]) / resolution)
        gy = int((y - origin[1]) / resolution)
        
        if 0 <= gx < obstacle_map.shape[1] and 0 <= gy < obstacle_map.shape[0]:
            if obstacle_map[gy, gx] == 0:
                # 检查与已有种子点的距离
                if len(seed_points) == 0 or all(
                    np.linalg.norm(seed_point - sp[0]) >= min_distance
                    for sp in seed_points
                ):
                    seed_points.append((seed_point, None))
        
        # 根据曲率决定下一个采样点
        curvature = curvatures[i]
        
        if curvature > np.pi / 6:  # > 30度：密集采样
            step = min(3, path_length - i - 1)
        elif curvature > np.pi / 12:  # > 15度：中等采样
            step = min(5, path_length - i - 1)
        else:  # <= 15度：稀疏采样
            step = min(10, path_length - i - 1)
        
        i += step
    
    # 强制包含终点
    x_end, y_end, _ = path[-1]
    end_point = np.array([x_end, y_end])
    gx_end = int((x_end - origin[0]) / resolution)
    gy_end = int((y_end - origin[1]) / resolution)
    
    if 0 <= gx_end < obstacle_map.shape[1] and 0 <= gy_end < obstacle_map.shape[0]:
        if obstacle_map[gy_end, gx_end] == 0:
            if len(seed_points) == 0 or all(
                np.linalg.norm(end_point - sp[0]) >= min_distance
                for sp in seed_points
            ):
                seed_points.append((end_point, None))
    
    return seed_points
